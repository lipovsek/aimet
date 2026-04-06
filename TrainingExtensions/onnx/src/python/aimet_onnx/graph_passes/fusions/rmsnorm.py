# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
RMSNormalization fusion pass for onnx-ir models
"""

import onnx_ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from .ir_utils import get_constant_singleton_value
from .fusion_registry import register_fusion, AIMET_SUPERGROUP_DOMAIN
from . import _patterns


AXIS_NAME = "axis"
EPS_NAME = "epsilon"


@register_fusion("RMSNormalization", pattern_idx=0)
@register_fusion("RMSNormalization", pattern_idx=1)
@register_fusion("RMSNormalization", pattern_idx=2)
class NonAffineRMSNormFusion(rewriter.RewriteRuleClassBase):
    """
    NonAffineRMSNormalization(x) = x / Sqrt(E(x**2) + ε)

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
    """

    def __init__(self, *args, pattern_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if pattern_idx not in {0, 1, 2}:
            raise ValueError(f"Invalid value for div_impl: {pattern_idx}")
        self.pattern_idx = pattern_idx

    # pylint: disable=arguments-differ
    def pattern(
        self,
        op: pattern.OpsetPatternBuilder,
        input_x: pattern.Var,
    ):
        return _patterns.non_affine_rms_normalize(
            op, input_x, EPS_NAME, AXIS_NAME, self.pattern_idx
        )

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        input_x: onnx_ir.Value,
        **kwargs,
    ) -> rewriter.pattern.MatchResult:
        match_result = pattern.MatchResult()
        axes: onnx_ir.Attr | onnx_ir.Value | None = kwargs.get(AXIS_NAME)
        epsilon: onnx_ir.Value | None = kwargs.get(EPS_NAME)

        axes_value = get_constant_singleton_value(axes)
        if axes is not None and axes_value is None:
            return match_result.fail(
                f"Axes attribute must be single element constant, got: {axes}"
            )

        eps = get_constant_singleton_value(epsilon)
        if eps is None or eps < 0:
            return match_result.fail(
                f"Epsilon must be a constant positive value, got {epsilon}"
            )

        return match_result

    # pylint: disable=arguments-differ
    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        input_x: onnx_ir.Value,
        **kwargs,
    ) -> onnx_ir.Value:
        axis = get_constant_singleton_value(kwargs.get(AXIS_NAME))
        axis = axis if axis is not None else -1
        output = op.RMSNormalization(
            input_x,
            axis=axis,
            epsilon=get_constant_singleton_value(kwargs.get(EPS_NAME)),
            _domain=AIMET_SUPERGROUP_DOMAIN,
        )
        return output


@register_fusion("RMSNormalization", pattern_idx=0)
@register_fusion("RMSNormalization", pattern_idx=1)
class RMSNormFusion(pattern.RewriteRuleClassBase):
    """
    Expected graph:

                x
                |
        RMSNormalization(x)
                |
               Mul

    """

    def __init__(self, *args, pattern_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if pattern_idx not in {0, 1}:
            raise ValueError(f"Invalid value for pattern_idx: {pattern_idx}")
        self.pattern_idx = pattern_idx

    # pylint: disable=arguments-differ
    def pattern(
        self,
        op: pattern.OpsetPatternBuilder,
        input_x: pattern.Var,
        scale: pattern.Var,
    ):
        return _patterns.rms_normalize(
            op, input_x, scale, EPS_NAME, AXIS_NAME, self.pattern_idx
        )

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        input_x: onnx_ir.Value,
        **kwargs,
    ) -> rewriter.pattern.MatchResult:
        match_result = pattern.MatchResult()
        axes: onnx_ir.Attr | None = kwargs.get(AXIS_NAME)
        epsilon: onnx_ir.Attr | None = kwargs.get(EPS_NAME)

        axes_value = get_constant_singleton_value(axes)
        if axes is not None and axes_value is None:
            return match_result.fail(
                f"Axes attribute must be single element constant, got: {axes}"
            )

        eps = get_constant_singleton_value(epsilon)
        if epsilon is None or eps < 0:
            return match_result.fail(
                f"Epsilon must be a constant positive value, got {epsilon}"
            )

        return match_result

    # pylint: disable=arguments-differ
    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        input_x: onnx_ir.Value,
        scale: onnx_ir.Value,
        **kwargs,
    ) -> onnx_ir.Value:
        epsilon = get_constant_singleton_value(kwargs.get(EPS_NAME))
        axis = get_constant_singleton_value(kwargs.get(AXIS_NAME))
        axis = axis if axis is not None else -1
        output = op.RMSNormalization(
            input_x, scale, axis=axis, epsilon=epsilon, _domain=AIMET_SUPERGROUP_DOMAIN
        )
        return output
