# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""holds common code for bias correction"""

import numpy as np
from scipy.stats import norm

from .defs import ActivationType
from .utils import AimetLogger
from .connected_graph.operation import Op

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

CONV_OP_TYPES = [
    "Conv1d",
    "Conv2D",
    "DepthwiseConv2dNative",
    "Conv",
    "ConvTranspose",
    "Conv3d",
]
LINEAR_OP_TYPES = ["Dense", "Gemm", "MatMul"]
BN_OP_TYPES = [
    "FusedBatchNormV3",
    "FusedBatchNorm",
    "BatchNormalization",
    "BatchNorm3d",
]


class ConvBnInfoType:
    """
    Type for hoding convs with bn info and activation types
    Activation types supported are Relu and Relu6
    """

    def __init__(
        self,
        input_bn=None,
        output_bn=None,
        in_activation_type: ActivationType = ActivationType.no_activation,
        out_activation_type: ActivationType = ActivationType.no_activation,
    ):
        """
        :param input_bn: Reference to Input BatchNorm to layer
        :param output_bn: Reference to Output BatchNorm to layer
        :param in_activation_type: Type of Activation
        :param out_activation_type: Type of Activation
        """

        self.input_bn = input_bn
        self.output_bn = output_bn
        self.in_activation_type = in_activation_type
        self.out_activation_type = out_activation_type


class ConvBnPatternHandler:
    """
    common handler for matched patterns for bias correction and batchnorm fold.
    """

    def __init__(self):
        self.conv_linears_with_bn_dict = {}

    def get_conv_linear_bn_info_dict(self):
        """
        returns the dictionary created
        :return: dictionary of convs/linears with bn and activation info
        """
        return self.conv_linears_with_bn_dict

    def __call__(self, *args, **kwargs):
        """
        custom pattern match handler that keeps a dictionary of convs/linears with bn and activation info.
        """

        _, op_subset = args

        bn_activation_info = ConvBnInfoType()

        activation_type = ActivationType.no_activation
        conv_op = None
        bn_op = None

        for op in op_subset:
            if op.type in CONV_OP_TYPES + LINEAR_OP_TYPES:
                conv_op = op
                op_key = get_op_dict_key(conv_op)
                if op_key in self.conv_linears_with_bn_dict:
                    bn_activation_info = self.conv_linears_with_bn_dict[op_key]
            elif op.type in BN_OP_TYPES:
                bn_op = op
            elif op.type in ["Relu6", "Clip"]:
                activation_type = ActivationType.relu6
            elif op.type in ["Relu"]:
                activation_type = ActivationType.relu

        if len(op_subset) >= 2:
            if op_subset[0].type in BN_OP_TYPES:
                bn_activation_info.input_bn = bn_op
                bn_activation_info.in_activation_type = activation_type
            # we do not match linear layers with preceding bn for bias correction
            elif op_subset[0].type in CONV_OP_TYPES + LINEAR_OP_TYPES:
                bn_activation_info.output_bn = bn_op
                bn_activation_info.out_activation_type = activation_type
            # in tf linear layer has two ops together [flatten/reshape -- dense] , check for len 3
            elif len(op_subset) >= 3 and op_subset[1].type in ["Dense"]:
                bn_activation_info.output_bn = bn_op
                bn_activation_info.out_activation_type = activation_type
        op_key = get_op_dict_key(conv_op)
        self.conv_linears_with_bn_dict[op_key] = bn_activation_info


def get_op_dict_key(op: Op):
    """
    Returns the object to be used as a key in the conv/linear BN dict.
    For torch and tensorflow models, returns op.get_module(). For onnx models, returns the original op.

    :param op: connected graph layer to be used as a dictionary key
    :return: object (op or op.get_module()) to be used as a key in the conv/linear BN dict
    """
    module = op.get_module()
    # ONNX NodeProto objects are not hashable, return the original Op object instead
    try:
        hash(module)
    except TypeError:
        return op

    return module


def empirical_bias_correction(
    reference_outputs: np.ndarray, quantized_outputs: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Empirical bias correction.

    :param quantized_outputs:
    :param reference_outputs:
    :param bias:
    :return: Updated bias
    """
    error = quantized_outputs - reference_outputs
    error = error.mean(3).mean(2).mean(0)
    _bias = bias - error
    return _bias


def analytical_bias_correction(
    fp_weight: np.ndarray,
    q_dq_weight: np.ndarray,
    bias: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    activation_type: ActivationType,
) -> np.ndarray:
    """
    Analytical bias correction.

    :param fp_weight:
    :param q_dq_weight:
    :param bias:
    :param beta:
    :param gamma:
    :param activation_type:
    :return: Updated bias
    """
    diff = q_dq_weight - fp_weight
    epsilon = diff.sum(3).sum(2)

    if activation_type == ActivationType.no_activation:
        e_x = beta
    elif activation_type == ActivationType.relu:
        e_x = beta * (1 - norm.cdf(-beta / gamma)) + gamma * norm.pdf(-beta / gamma)
    elif activation_type == ActivationType.relu6:
        b = 6
        z = norm.pdf(-beta / gamma) - norm.pdf((b - beta) / gamma)
        Z = norm.cdf((b - beta) / gamma) - norm.cdf(-beta / gamma)
        e_x = gamma * z + beta * Z + b * (1 - norm.cdf((b - beta) / gamma))
    else:
        raise ValueError("Unsupported activation type: ", activation_type)

    if epsilon.shape[1] == 1:
        ep = epsilon.reshape(epsilon.shape[0])
        error = np.multiply(ep, e_x)
    else:
        error = np.matmul(epsilon, e_x)

    _bias = bias - error
    return _bias
