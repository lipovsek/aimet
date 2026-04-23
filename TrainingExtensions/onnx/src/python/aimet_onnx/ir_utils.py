# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-ir related utility functions"""

import onnx_ir
from aimet_onnx.graph_passes.fusions.ir_utils import *  # pylint: disable=wildcard-import, unused-wildcard-import


def remove_aimet_quantizers(model: onnx_ir.Model):
    quant_nodes = [
        node for node in model.graph.all_nodes() if node.op_type == "QcQuantizeOp"
    ]
    quant_output_to_input = {node.outputs[0]: node.inputs[0] for node in quant_nodes}
    model.graph.remove(quant_nodes)

    for node in model.graph.all_nodes():
        for i, input_name in enumerate(node.inputs):
            if input_name in quant_output_to_input:
                node.replace_input_with(i, quant_output_to_input[input_name])

    for i, output in enumerate(model.graph.outputs):
        if output in quant_output_to_input:
            output.name = quant_output_to_input[output].name
