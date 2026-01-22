# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities for ONNX Connected Graph"""

from typing import Dict, List
import onnx
from packaging import version

from aimet_onnx.meta.connectedgraph import ConnectedGraph

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto


ActivationTypes = ["Relu", "Clip", "Sigmoid", "Tanh", "PRelu", "Softmax"]


def get_op_given_param_name(connected_graph: ConnectedGraph, param_name: str):
    """
    Gets op for a given param name

    :param connected_graph: Connected graph
    :param param_name: Name of the parameter
    """
    ops = connected_graph.get_all_ops()
    for op in ops.values():
        if op.parameters:
            for param, _ in op.parameters.values():
                if param.name == param_name:
                    return op
    return None


def get_param_shape_using_connected_graph(
    connected_graph: ConnectedGraph, param_name: str
):
    """
    Gets param shape given param name

    :param connected_graph: Connected graph
    :param param_name: Name of the parameter
    """
    ops = connected_graph.get_all_ops()
    for op in ops.values():
        if op.parameters:
            for param, _ in op.parameters.values():
                if param.name == param_name:
                    return param.shape
    return None


def get_module_act_func_pair(graph: ConnectedGraph) -> Dict[str, str]:
    """
    For given graph, returns dictionary of module to immediate following activation function if present

    Activation functions should be defined as nn.Modules in model and not as functional in the forward pass.

    :param graph: ConnectedGraph object
    :return: Dictionary of module name to activation function name
    """
    # Maps module to next following activation function else None
    module_act_func_pair = {}

    # Get all the ops
    all_ops = graph.get_all_ops()

    for op in all_ops.values():
        if op.output_ops:
            # Get the next op
            next_op = op.output_ops[0]
            # Get the appropriate activation function
            if next_op.type in ActivationTypes:
                module_act_func_pair[op.name] = next_op.type

    return module_act_func_pair
