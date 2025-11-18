# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""ONNX Code to fold batch-norm layers"""

from typing import Dict, List, Tuple
import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version

from aimet_onnx.common.batch_norm_fold import batch_norm_fold
from aimet_onnx.common.bias_correction import ConvBnPatternHandler
from aimet_onnx.common.graph_pattern_matcher import PatternType
from aimet_onnx.common.graph_searcher import GraphSearcher
from aimet_onnx.common.connected_graph.connectedgraph_utils import get_ordered_ops
from aimet_onnx.common.utils import AimetLogger

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.connectedgraph import (
    WEIGHT_INDEX,
    BIAS_INDEX,
    RUNNING_MEAN_INDEX,
    RUNNING_VAR_INDEX,
)
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import (
    get_node_attribute,
    remove_node,
    transpose_tensor,
    ParamUtils,
    retrieve_constant_input,
)

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import NodeProto, ModelProto
else:
    from onnx.onnx_pb import NodeProto, ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BatchNormFolding)

ConvType = ["Conv", "ConvTranspose"]
LinearType = ["Gemm", "MatMul"]
BatchNormType = ["BatchNormalization"]


class BNLayer:
    """Captures beta and gamma parameter for BatchNorm layers to be used during High Bias absorption"""

    def __init__(self, bn_layer=None, gamma=None, beta=None):
        self.bn_layer = bn_layer
        self.gamma = gamma
        self.beta = beta


def _find_conv_bn_pairs(connected_graph: ConnectedGraph) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param connected_graph: ConnectedGraph object.
    :return: dictionary of conv/linear Op with associated bn op / activation info
    """

    # initialize all patterns to be matched and associated call back functions
    patterns_with_callbacks = []
    layer_select_handler = ConvBnPatternHandler()
    preceding_linear_op_types = ["Flatten", "Reshape"]

    # Linear layer combinations
    for linear_op in LinearType:
        for preceding_linear_op_type in preceding_linear_op_types:
            # BN -> Linear
            patterns_with_callbacks.append(
                PatternType(
                    pattern=["BatchNormalization", preceding_linear_op_type, linear_op],
                    action=layer_select_handler,
                )
            )

    for op_type in ConvType + LinearType:
        patterns_with_callbacks.append(
            PatternType(
                pattern=["BatchNormalization", op_type], action=layer_select_handler
            )
        )
        patterns_with_callbacks.append(
            PatternType(
                pattern=[op_type, "BatchNormalization"], action=layer_select_handler
            )
        )

    # create graph searcher instance with connected graph and patterns to search
    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)

    # get all conv/linear and bn info
    graph_searcher.find_all_patterns_in_graph_apply_actions()
    convs_bn_activation_dict = layer_select_handler.get_conv_linear_bn_info_dict()

    return convs_bn_activation_dict


def find_all_batch_norms_to_fold(
    connected_graph: ConnectedGraph,
) -> Tuple[List[Tuple[NodeProto, NodeProto]], List[Tuple[NodeProto, NodeProto]]]:
    """
    Find all possible batch norm layers that can be folded. Returns a list of pairs such that (bn, layer)
    means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
    :param connected_graph: connected graph model to search
    :return: A list of (layer, bn) pairs and a list of (bn, layer) pairs,
             where `bn` can be folded into to `layer`.
    """
    conv_linear_bn_activation_info_dict = _find_conv_bn_pairs(connected_graph)
    model = connected_graph.model
    # To mark BN's already picked for backward folding
    bn_picked_for_folding = set()

    ordered_conv_fc_nodes = get_ordered_conv_linears(connected_graph)

    conv_bn_pairs = []
    # Backward fold is given priority over Forward fold
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict:
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.output_bn and bn_info.output_bn not in bn_picked_for_folding:
                if is_valid_bn_fold(node.get_module(), model, True):
                    conv_bn_pairs.append(
                        (node.get_module(), bn_info.output_bn.get_module())
                    )
                    bn_picked_for_folding.add(bn_info.output_bn)
                else:
                    logger.info(
                        "...... invalid combination to fold %s",
                        [node.name, bn_info.output_bn.name],
                    )

    bn_conv_pairs = []
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict:
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.input_bn and bn_info.input_bn not in bn_picked_for_folding:
                if is_valid_bn_fold(node.get_module(), model, False):
                    bn_conv_pairs.append(
                        (bn_info.input_bn.get_module(), node.get_module())
                    )
                    bn_picked_for_folding.add(bn_info.input_bn)
                else:
                    logger.info(
                        "...... invalid combination to fold %s",
                        [bn_info.input_bn.name, node.name],
                    )

    return conv_bn_pairs, bn_conv_pairs


def get_ordered_conv_linears(conn_graph: ConnectedGraph) -> List[Op]:
    """
    helper to select a list of candidate layers for BatchNorm folding
    :param conn_graph: connected graph to search
    :return: List of conv/linear layers
    """
    # get ordered operations list from the connected graph
    list_of_ordered_ops = get_ordered_ops(conn_graph.starting_ops)

    # look for conv/linear layers
    ordered_convs = []
    for op in list_of_ordered_ops:
        if op.type in ConvType + LinearType:
            ordered_convs.append(op)
    return ordered_convs


def is_valid_bn_fold(
    conv_linear: NodeProto, model: ModelProto, fold_backward: bool
) -> bool:
    """
    Determine if a given layer can successfully absorb a BatchNorm given the layer type and parameters
    :param conv_linear: The Conv/Linear layer to fold a BatchNorm into.
    :param model: The model to which the Conv/Linear layer belongs.
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    :return: True if a BatchNorm layer can be folded without causing output error.
    """
    valid = True
    if conv_linear.op_type in LinearType:
        # Check if this is actually a fully connected layer or a dynamic matmul
        w = retrieve_constant_input(conv_linear, model, WEIGHT_INDEX)[0]
        if w is None:
            valid = False
    if not fold_backward:
        # Cannot fold BN -> Conv with padding. AIMET does not support forward folding to grouped or DW Conv
        if conv_linear.op_type == "Conv":
            valid &= all(item == 0 for item in get_node_attribute(conv_linear, "pads"))
            valid &= get_node_attribute(conv_linear, "group") == 1
        # AIMET does not support forward folding to ConvTranspose
        elif conv_linear.op_type == "ConvTranspose":
            valid = False
    else:
        # AIMET does not support backwards folding to grouped ConvTranspose
        if conv_linear.op_type == "ConvTranspose":
            if get_node_attribute(conv_linear, "group") is not None:
                valid &= get_node_attribute(conv_linear, "group") in (
                    1,
                    get_input_output_channels(conv_linear, model)[0],
                )
    return valid


def fold_all_batch_norms_to_weight(model: ModelProto) -> Tuple[List, List]:
    """
    Fold all possible batch_norm layers in a model into the weight of the corresponding conv layers

    :param model: onnx Model to perform BN fold on
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    if isinstance(model, ONNXModel):
        model = model.model
    connected_graph = ConnectedGraph(model)
    model = connected_graph.model
    conv_bn_pairs, bn_conv_pairs = find_all_batch_norms_to_fold(connected_graph)
    conv_bns = []
    bn_convs = []
    for conv, bn in conv_bn_pairs:
        bn_layer = _fold_to_weight(model, conv, bn, True)
        conv_bns.append((conv, bn_layer))
        remove_node(bn, model.graph)

    for bn, conv in bn_conv_pairs:
        bn_layer = _fold_to_weight(model, conv, bn, False)
        bn_convs.append((conv, bn_layer))
        remove_node(bn, model.graph)

    _update_standalone_batchnorm_ops(model)

    return conv_bns, bn_convs


def _fold_to_weight(
    model: ModelProto, conv_linear: NodeProto, bn: NodeProto, fold_backward: bool
):
    """
    Fold BatchNorm into the weight and bias of the given layer.

    :param model: onnx model to which the conv/bn pair belong
    :param conv_linear: Conv or linear layer to fold BN into.
    :param bn: BatchNorm to fold.
    :param fold_backward: True if the BatchNorm comes after the Conv
    """
    # Must convert MatMul layers to Gemm to allow bias
    if conv_linear.op_type == "MatMul":
        _matmul_to_gemm(conv_linear, model)

    weight = ParamUtils.get_param(model, conv_linear, WEIGHT_INDEX)
    bias = ParamUtils.get_param(model, conv_linear, BIAS_INDEX)
    groups = get_node_attribute(conv_linear, "group")
    if not groups:
        groups = 1
    _, num_out_channels = get_input_output_channels(conv_linear, model)

    # If layer doesn't have bias, create a bias initializer and add it to the model, then retrieve it
    if not bias:
        bias_data = np.zeros(num_out_channels)
        bias_name = conv_linear.name + ".bias"
        bias = numpy_helper.from_array(bias_data.astype(np.float32), name=bias_name)
        model.graph.initializer.append(bias)
        conv_linear.input.append(bias_name)
        bias = ParamUtils.get_param(model, conv_linear, BIAS_INDEX)

    weight_np = numpy_helper.to_array(weight)
    weight_np = np.expand_dims(weight_np, axis=tuple(range(weight_np.ndim, 4)))
    bias_np = numpy_helper.to_array(bias)

    # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
    # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
    # if conv_linear.type == "ConvTranspose" and conv_linear groups == 1:
    if conv_linear.op_type == "ConvTranspose" and groups == 1:
        weight_np = weight_np.transpose(1, 0, 2, 3)
    # Gemm layers may or may not need to have weights transposed depending on value of transB attribute
    elif conv_linear.op_type in LinearType and not get_node_attribute(
        conv_linear, "transB"
    ):
        weight_np = weight_np.transpose(1, 0, 2, 3)

    gamma = ParamUtils.get_param(model, bn, WEIGHT_INDEX)
    beta = ParamUtils.get_param(model, bn, BIAS_INDEX)
    mu = ParamUtils.get_param(model, bn, RUNNING_MEAN_INDEX)
    running_var = ParamUtils.get_param(model, bn, RUNNING_VAR_INDEX)

    gamma_np = numpy_helper.to_array(gamma)
    beta_np = numpy_helper.to_array(beta)
    mu_np = numpy_helper.to_array(mu)
    epsilon = get_node_attribute(bn, "epsilon") or 1e-5
    sigma_np = np.sqrt(numpy_helper.to_array(running_var) + epsilon)

    # In the case of BatchNorm2d -> Flatten -> Gemm, must resize the BN parameters to the Gemm input feature length
    channels = weight_np.shape[0] if fold_backward else weight_np.shape[1]
    gamma_np = gamma_np.repeat(channels / gamma_np.size)
    beta_np = beta_np.repeat(channels / beta_np.size)
    mu_np = mu_np.repeat(channels / mu_np.size)
    sigma_np = sigma_np.repeat(channels / sigma_np.size)

    weight_np, bias_np = batch_norm_fold(
        weight_np, bias_np, gamma_np, beta_np, mu_np, sigma_np, fold_backward
    )

    # Transpose weight back to original configuration
    if conv_linear.op_type == "ConvTranspose" and groups == 1:
        weight_np = weight_np.transpose(1, 0, 2, 3)
    elif conv_linear.op_type in LinearType and not get_node_attribute(
        conv_linear, "transB"
    ):
        weight_np = weight_np.transpose(1, 0, 2, 3)

    weight_np = weight_np.astype(onnx.helper.tensor_dtype_to_np_dtype(weight.data_type))
    bias_np = bias_np.astype(onnx.helper.tensor_dtype_to_np_dtype(bias.data_type))
    weight.raw_data = weight_np.tobytes()
    bias.raw_data = bias_np.tobytes()

    return BNLayer(bn, gamma_np, beta_np)


def _matmul_to_gemm(node: NodeProto, model: ModelProto):
    """
    Convert MatMul node to Gemm and initialize bias to zeros

    :param node: MatMul node to convert to Gemm
    :param model: model to which the node belongs
    """
    assert node.op_type == "MatMul"

    weight, transposed = retrieve_constant_input(node, model, WEIGHT_INDEX)
    if transposed:
        node.input[WEIGHT_INDEX] = weight.name
        model.graph.initializer.remove(weight)
        weight = transpose_tensor(weight, (1, 0))
        model.graph.initializer.append(weight)
    node.op_type = "Gemm"
    node.name = node.name.replace("MatMul", "Gemm")
    # Create bias vector for Gemm operation
    bias_name = node.name + ".bias"
    bias_data = np.zeros(weight.dims[1])
    bias = numpy_helper.from_array(bias_data.astype(np.float32), name=bias_name)
    model.graph.initializer.append(bias)
    node.input.append(bias_name)


def _get_input_output_channel_axes(node: NodeProto) -> Tuple[int, int]:
    if node.op_type == "Conv":
        return 1, 0
    elif node.op_type == "ConvTranspose":
        return 0, 1
    elif node.op_type == "Gemm":
        transB = get_node_attribute(node, "transB")
        if transB == 1:
            return 1, 0
        else:
            return 0, 1
    else:
        raise RuntimeError


def get_input_output_channels(node: NodeProto, model: ModelProto) -> Tuple[int, int]:
    """
    Find the input and output channels of a given layer.
    :param node: The node to find the input/output channels of
    :param model: The onnx model to which the layers belong
    :return: Tuple of (num channels in, num channels out)
    """
    weight = ParamUtils.get_param(model, node, WEIGHT_INDEX)
    in_axis, out_axis = _get_input_output_channel_axes(node)
    groups = get_node_attribute(node, "group")
    # If group atttribute does not exist in the node,then default is 1
    if not groups:
        groups = 1
    if node.op_type == "Conv":
        num_in_channels = weight.dims[in_axis] * groups
        num_out_channels = weight.dims[out_axis]
    elif node.op_type == "ConvTranspose":
        num_in_channels = weight.dims[in_axis]
        num_out_channels = weight.dims[out_axis] * groups
    elif node.op_type == "Gemm":
        transB = get_node_attribute(node, "transB")
        if transB == 1:
            num_out_channels = weight.dims[out_axis]
            num_in_channels = weight.dims[in_axis]
        else:
            num_out_channels = weight.dims[out_axis]
            num_in_channels = weight.dims[in_axis]
    else:
        num_out_channels = None
        num_in_channels = None
    return num_in_channels, num_out_channels


# pylint: disable=too-many-locals
def _update_standalone_batchnorm_ops(model: ModelProto):
    """
    Update weight and bias of standalone batchnorm ops in the model.
    :param model: onnx Model for which batchnorm parameters are to be updated.
    """

    for node in model.graph.node:
        if node.op_type in BatchNormType:
            # get parameter names and indices
            weight_name, bias_name, running_mean_name, running_var_name = node.input[1:]
            init_w, init_b, init_rm, init_rv = [
                ParamUtils.get_param(model, node, idx) for idx in range(1, 5)
            ]

            attr = [item for item in node.attribute if item.name == "epsilon"]
            if not attr:
                attr = onnx.helper.make_attribute(
                    "epsilon", 1e-5
                )  # Default epsilon value
                node.attribute.append(attr)
            else:
                attr = attr[0]

            epsilon = attr.f
            tensor_w = numpy_helper.to_array(init_w)
            tensor_b = numpy_helper.to_array(init_b)
            tensor_rm = numpy_helper.to_array(init_rm)
            tensor_rv = numpy_helper.to_array(init_rv)

            # update values
            inv_sigma = np.reciprocal(np.sqrt(tensor_rv + epsilon))
            tensor_w = tensor_w * inv_sigma
            tensor_b = tensor_b - tensor_rm * tensor_w
            tensor_rm = np.zeros(tensor_w.shape, tensor_w.dtype)
            tensor_rv = np.ones(tensor_w.shape, tensor_w.dtype)
            attr.f = 0.0

            init_w_ = numpy_helper.from_array(tensor_w, weight_name)
            init_b_ = numpy_helper.from_array(tensor_b, bias_name)
            init_rm_ = numpy_helper.from_array(tensor_rm, running_mean_name)
            init_rv_ = numpy_helper.from_array(tensor_rv, running_var_name)

            # update initializers
            init_w.CopyFrom(init_w_)
            init_b.CopyFrom(init_b_)
            init_rm.CopyFrom(init_rm_)
            init_rv.CopyFrom(init_rv_)
