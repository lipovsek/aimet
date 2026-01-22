# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


from aimet_onnx.meta.operations import Op
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from typing import Dict, List


def assert_on_const_quantizers(
    ops: List[Op], qc_quantize_op_dict: Dict[str, QcQuantizeOp], enabled: bool = False
):
    """
    Assert on all constant inputs for provided list of ops with given condition

    Args:
        ops (List[Op]): List of ops to check constant inputs
        qc_quantize_op_dict (Dict[str, QcQuantizeOp]): Global dictionary of quantizer names to Quantize op.
        enabled (bool, optional): Condition to check. Defaults to False.
    """
    for op in ops:
        for op_input in op.inputs:
            if op_input.is_const and op_input.name in qc_quantize_op_dict:
                assert qc_quantize_op_dict[op_input.name].enabled == enabled


def assert_on_output_quantizers(
    ops: List[Op], qc_quantize_op_dict: Dict[str, QcQuantizeOp], enabled: bool = False
):
    """
    Assert on all output quantizers for provided list of ops with given condition

    Args:
        ops (List[Op]): List of ops to check output quantizers
        qc_quantize_op_dict (Dict[str, QcQuantizeOp]): Global dictionary of quantizer names to Quantize op.
        enabled (bool, optional): Condition to check. Defaults to False.
    """
    for op in ops:
        for output in op.outputs:
            if output.name in qc_quantize_op_dict:
                assert qc_quantize_op_dict[output.name].enabled == enabled
