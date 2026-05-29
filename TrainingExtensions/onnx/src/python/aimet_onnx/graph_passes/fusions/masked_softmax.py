# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
import onnx_ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from aimet_onnx.graph_passes.fusions.ir_utils import get_constant_singleton_value
from aimet_onnx.graph_passes.fusions.fusion_registry import (
    register_fusion,
    AIMET_SUPERGROUP_DOMAIN,
)


@register_fusion(name="MaskedSoftmax")
class MaskedSoftmax(pattern.RewriteRuleClassBase):
    """
    Softmax(
        Where(mask, x, ReduceMin(x, axis=-1) + B),
        axis=-1,
    )
      where
        - x: 4D
        - B: scalar constant <=-20
    """

    # pylint: disable=arguments-differ
    def pattern(
        self,
        op: pattern.OpsetPatternBuilder,
        x: pattern.Var,
        mask: pattern.Var,
        mask_val: pattern.Var,
        reducemin_axes: pattern.Var,
    ):
        softmax_axis = pattern.AttrVar("softmax_axis")

        return op.Softmax(
            op.Where(
                mask,
                x,
                op.ReduceMin(x, reducemin_axes) + mask_val,
            ),
            axis=softmax_axis,
        )

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        x: onnx_ir.Value,
        mask: onnx_ir.Value,
        mask_val: onnx_ir.Value,
        reducemin_axes: onnx_ir.Value,
        *,
        softmax_axis: onnx_ir.Attr,
    ) -> rewriter.pattern.MatchResult:
        """
        Validates that a matched pattern satisfies additional constraints on MaskedSoftmax op.
        """
        match_result = pattern.MatchResult()

        if not (x.shape and len(x.shape) == 4):
            return match_result.fail(f"Input must be 4D tensor, got {x.shape}")

        if get_constant_singleton_value(softmax_axis) not in (-1, 3):
            return match_result.fail(
                f"Softmax axis must be a constant with value -1 or 3, got {softmax_axis}"
            )

        if get_constant_singleton_value(reducemin_axes) not in (-1, 3):
            return match_result.fail(
                f"ReduceMin axis must be a constant with value -1 or 3, got {reducemin_axes}"
            )

        if get_constant_singleton_value(mask_val) is None:
            return match_result.fail(
                f"Mask value must be a constant scalar, got {mask_val}"
            )

        if get_constant_singleton_value(mask_val) > -20:
            return match_result.fail(
                f"Mask value must be a constant with value <= -20, got {mask_val}"
            )

        return match_result

    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        x: onnx_ir.Value,
        mask: onnx_ir.Value,
        mask_val: onnx_ir.Value,
        reducemin_axes: onnx_ir.Value,
        *,
        softmax_axis: onnx_ir.Attr | onnx_ir.Value,
    ) -> onnx_ir.Value:
        """
        Defines the fused replacement for the matched decomposed pattern.
        """
        return op.MaskedSoftmax(
            x,
            mask,
            axis=-1,
            _domain=AIMET_SUPERGROUP_DOMAIN,
        )
