# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
from typing import List, Tuple
from aimet_onnx.graph_passes.pass_registry import PASS_REGISTRY
from aimet_onnx.quantsim import QuantizationSimModel

OP_TYPES_IN_BLOCKS = ["Conv", "MatMul", "Gemm"]
PASS_TO_RUN = "DecoderBlock"


def get_conv_linear_layers_decoder_block(
    quantsim: QuantizationSimModel, decoder_blocks_end_points: List[Tuple]
) -> List[Tuple]:
    """
    Gets Conv or linear layers in a decoder block
    :param quantsim: quantization simulator
    :param decoder_blocks_end_points: end points of the decoder block
    """
    all_ops = quantsim.connected_graph.ordered_ops
    layers_in_each_decoder_block = []
    op_name_to_index = {}
    for index, op in enumerate(all_ops):
        op_name_to_index[op.name] = index

    for i in range(len(decoder_blocks_end_points)):
        start, end = decoder_blocks_end_points[i]
        if start.name in op_name_to_index and end.name in op_name_to_index:
            start_index = op_name_to_index[start.name]
            end_index = op_name_to_index[end.name]
            decoder_ops = []
            for j in range(start_index, end_index):
                op = all_ops[j]
                if op.type in OP_TYPES_IN_BLOCKS:
                    decoder_ops.append(op)
            layers_in_each_decoder_block.append(decoder_ops)
    return layers_in_each_decoder_block


def get_all_layers_per_decoder_block(
    quantsim: QuantizationSimModel,
    decoder_blocks_end_points: List[Tuple],
    block_index: int,
) -> List[Tuple]:
    """
    Returns all the layers between decoder block boundaries
    """
    all_ops = quantsim.connected_graph.ordered_ops
    op_name_to_index = {op.name: index for index, op in enumerate(all_ops)}
    return all_ops[
        op_name_to_index[
            str(decoder_blocks_end_points[block_index][0])
        ] : op_name_to_index[str(decoder_blocks_end_points[block_index][1])] + 1
    ]


def get_decoder_blocks_end_points(quantsim: QuantizationSimModel) -> List[Tuple]:
    """
    Gets end points of the decoder blocks
    :param quantsim: quantization simulator
    """
    if PASS_TO_RUN in PASS_REGISTRY:
        graph_pass_obj = PASS_REGISTRY[PASS_TO_RUN]
        graph_pass_obj(
            quantsim.model.model, quantsim.connected_graph, quantsim.qc_quantize_op_dict
        )
        decoder_blocks_end_points = graph_pass_obj.decoder_blocks
        return decoder_blocks_end_points
    raise ValueError(f"Graph pass requested but not found: {PASS_TO_RUN}")
