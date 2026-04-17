# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

from typing import List
from aimet_onnx.common.connected_graph.operation import Op
from aimet_onnx.graph_passes.graph_pass import SupergroupGraphPass
from aimet_onnx.graph_passes.pass_registry import register_pass
from aimet_onnx.graph_passes.passes.common_patterns import match_rms_norm_pattern
from aimet_onnx.utils import ModelProto


@register_pass("RMSNormalization")
class RMSNormalization(SupergroupGraphPass):
    """
    Disable output quantizers for RMSNormalization intermediate ops:

    RMSNormalization(x) = x / Sqrt(E(x**2) + ε) * γ

    Expected graph:
    Version 1: With x * div ( 1 / denominator )
                x
            +---+---+
            |       |
    Mul or Pow(x, 2)|
            |       |
        ReduceMean  |
            |       |
            Add     |
            |       |
            Sqrt    |
        1   |       |
        +-- Div     |
            |       |
            +---+---+
                Mul
                |
                Mul (if elementwise_affine=True)

    Version 2: With x * div ( 1 / denominator )
                x
            +---+---+
            |       |
            |       Mul or Pow(x, 2)
            |       |
            |       ReduceMean
            |       |
            |       Add
            |       |
            |       Sqrt
            |       |
            +---+---+
                Div
                |
                Mul (if elementwise_affine=True)
    """

    # pylint: disable=too-many-branches, too-many-return-statements
    def match_pattern(self, op: Op, model: ModelProto) -> List[Op]:
        """
        Match RMSNormalization pattern and collect ops to disable output quantizers
        """
        all_ops = match_rms_norm_pattern(op, model)
        if not all_ops:
            return []

        # Check if weights are present, skipping any Cast ops (e.g. from
        # dtype casting in RMSNorm implementations like Qwen2RMSNorm which
        # casts to float32 for numerical stability then back).
        elementwise_affine = False
        next_op = all_ops[-1]
        cast_ops = []
        while len(next_op.output_ops) == 1 and next_op.output_ops[0].type == "Cast":
            cast_ops.append(next_op.output_ops[0])
            next_op = next_op.output_ops[0]
        if len(next_op.output_ops) == 1 and next_op.output_ops[0].type == "Mul":
            elementwise_affine = True
            all_ops.extend(cast_ops)
            all_ops.append(next_op.output_ops[0])

        # Disable output quantizers for all the intermediate outputs
        self.disable_output_quantizers(all_ops[:-1])
        # Disable all constant quantizers except weights
        self.disable_const_quantizers(all_ops[:-1] if elementwise_affine else all_ops)
        return all_ops
