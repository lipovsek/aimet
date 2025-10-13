# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)
import onnx
from onnx import numpy_helper
import os
from onnx.utils import extract_model
from onnx2torch import convert
from aimet_onnx.experimental.adascale.onnx2torch_ext import *  # pylint: disable=wildcard-import, unused-wildcard-import
from onnx2torch.onnx_graph import OnnxGraph
import tempfile

filter_op = ["MatMul", "Conv"]


def _get_onnx_subgraph(onnx_fp_model_path, block_input_output_names, block_model_path):
    """
    Given a onnx block end points get onnx subgraph
    """
    block_model_path = os.path.join(block_model_path, "block_fp32.onnx")
    block_input_names, block_output_names = block_input_output_names
    try:
        extract_model(
            onnx_fp_model_path,
            block_model_path,
            block_input_names,
            block_output_names,
        )
        block_fp32_model = onnx.load(block_model_path)
        return block_fp32_model
    except Exception:
        raise RuntimeError(  # pylint: disable=raise-missing-from
            f"Unable to extract onnx subgraph for given block input/output {block_input_output_names}"
        )


def _get_onnx_block_info(onnx_subgraph):
    """
    For an onnx subgraph get onnx param name from initializer list map
    """
    graph = onnx_subgraph.graph
    name_to_node_filtered = {n.name: n for n in graph.node if n.op_type in filter_op}
    initializer_name_to_index_map = {
        init.name: idx for idx, init in enumerate(graph.initializer)
    }
    node_name_to_onnx_param = {}
    for node in name_to_node_filtered.values():
        for edge in node.input:
            if edge in initializer_name_to_index_map and "bias" not in edge:
                # Bias will not be updated so we donot need to keep track of bias
                node_name_to_onnx_param[OnnxGraph.generate_node_name(node)] = edge
    return node_name_to_onnx_param


def get_pt_block(onnx_model_path: str, block_input_output_names: tuple):
    """
    Given a onnx block end points get a pytorch block
    """
    with tempfile.TemporaryDirectory() as tempdir:
        onnx_block = _get_onnx_subgraph(
            onnx_model_path, block_input_output_names, tempdir
        )
        param_map = _get_onnx_block_info(onnx_block)
        return convert(onnx_block), param_map


def copy_pt_weights_to_onnx(
    pt_block, onnx_model, param_map, initializer_name_to_index_map
):
    """
    Given a pt_block with adascale params computed, copy the params to onnx model
    """
    for name, module in pt_block.named_modules():
        if param_map.get(name) is None:
            continue
        pytorch_weight = module.weight.detach().cpu().numpy()
        onnx_tensor_name = param_map[name]
        onnx_param_tensor = numpy_helper.to_array(
            onnx_model.graph.initializer[
                initializer_name_to_index_map[onnx_tensor_name]
            ]
        )
        # For conv transpose is not required
        if pytorch_weight.shape != onnx_param_tensor.shape:
            pytorch_weight = pytorch_weight.T
        if pytorch_weight.shape != onnx_param_tensor.shape:
            raise ValueError(
                f"pt param shape {pytorch_weight.shape} did not match onnx shape {onnx_param_tensor.shape}"
            )
        if not (pytorch_weight == onnx_param_tensor).all():
            onnx_model.graph.initializer[
                initializer_name_to_index_map[onnx_tensor_name]
            ].CopyFrom(numpy_helper.from_array(pytorch_weight, onnx_tensor_name))
            _logger.info(
                "Copy from PyTorch to ONNX: torch : %s  onnx param : %s",
                name,
                onnx_tensor_name,
            )
