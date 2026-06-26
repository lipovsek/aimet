# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from typing import List
from aimet_onnx.common.connected_graph.operation import Op
from aimet_onnx.utils import ModelProto

from aimet_onnx.graph_passes.utils import (
    check_consecutive_ops,
    is_constant_scalar,
    match_pow_2_pattern,
    match_a_div_b_pattern,
)


def match_rms_norm_pattern(op: Op, model: ModelProto) -> List[Op]:
    """Common pattern for RMSNormalization which can be re-used"""
    if op.type == "RMSNormalization":
        return [op]

    # Match Mul(x, x) or Pow(x, 2)
    match = match_pow_2_pattern(op, model)
    if not match or len(op.output_ops) != 1:
        return []

    # E(Pow(x, 2)) + ε
    match, denominator_ops = check_consecutive_ops(
        op.output_ops[0],
        ["ReduceMean", "Add"],
        validate_last_op_consumers=False,
    )
    if not match:
        return []

    all_ops = [op] + denominator_ops
    add_op = all_ops[-1]

    if len(add_op.output_ops) != 1:
        return []

    next_op = add_op.output_ops[0]

    # Sqrt form: Div(x, Sqrt(E(Pow(x, 2)) + ε))
    if next_op.type == "Sqrt" and len(next_op.output_ops) == 1:
        div_ops = match_a_div_b_pattern(op.inputs[0], next_op.outputs[0], model)
        if div_ops:
            return all_ops + [next_op] + div_ops

    # rsqrt form: Mul(x, Pow(E(Pow(x, 2)) + ε, -0.5))
    if (
        next_op.type == "Pow"
        and is_constant_scalar(model, next_op.inputs[1], -0.5)
        and len(next_op.output_ops) == 1
    ):
        mul_op = next_op.output_ops[0]
        if mul_op.type == "Mul" and op.inputs[0] in (
            mul_op.inputs[0],
            mul_op.inputs[1],
        ):
            return all_ops + [next_op, mul_op]

    return []
