# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LayerNormalization fusion pass for ONNX models"""

import onnx_ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from .ir_utils import get_constant_singleton_value
from .fusion_registry import register_fusion, AIMET_SUPERGROUP_DOMAIN
from . import _patterns

_EPS_NAME = "epsilon"
_AXES_NAME = "axes"
_AXIS_NAME = "axis"


@register_fusion(name="LayerNormalization")
class LayerNormFusion(pattern.RewriteRuleClassBase):
    """
    Fuses decomposed LayerNormalization pattern into a single node.

    Implements pattern matching and replacement for the decomposed LayerNormalization operation:

    LayerNormalization(x, scale, bias, epsilon) = (x - E[x]) / sqrt(Var[x] + epsilon) * scale + bias

    Expected decomposed graph pattern:
                    x
                +---+---+
                |       |
            ReduceMean  |
                |       |
                +---+---+
                    Sub
                +---+---+
                |       |
                Pow(2)  |
                |       |
            ReduceMean  |
                |       |
                Add     |
                |       |
                Sqrt    |
                |       |
                +---+---+
                    Div
                    |
                    Mul
                    |
                    Add

    The pattern is replaced with a single function call node in the
    'aimet.supergroup' domain.
    """

    # pylint: disable=arguments-differ
    def pattern(
        self,
        op: pattern.OpsetPatternBuilder,
        input_x: pattern.Var,
        scale: pattern.Var,
        bias: pattern.Var,
    ):
        """
        Defines the decomposed LayerNormalization pattern to match.
        """
        axes_attr = pattern.AttrVar(_AXES_NAME)

        # E[x]
        mean = op.ReduceMean(input_x, axes=axes_attr)

        # x - E[x]
        centered = input_x - mean

        # Note: same attribute cannot match both ReduceMean "axes" attr and RMSNormalization "axis" attr
        scaled = pattern.OrValue(
            [
                _patterns.rms_normalize(op, centered, scale, _EPS_NAME, _AXIS_NAME),
                op.RMSNormalization(
                    centered,
                    scale,
                    epsilon=pattern.AttrVar(_EPS_NAME),
                    axis=pattern.AttrVar(_AXIS_NAME),
                    _domain=AIMET_SUPERGROUP_DOMAIN,
                ),
            ]
        )

        return scaled + bias

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        input_x: onnx_ir.Value,
        scale: onnx_ir.Value,
        bias: onnx_ir.Value,
        **kwargs,
    ) -> rewriter.pattern.MatchResult:
        """
        Validates that a matched pattern satisfies additional constraints on LayerNormalization ops.
        """
        match_result = pattern.MatchResult()
        epsilon: onnx_ir.Attr | onnx_ir.Value | None = kwargs.get(_EPS_NAME)
        axis_attr: onnx_ir.Attr | onnx_ir.Value | None = kwargs.get(_AXIS_NAME)
        axes_attr: onnx_ir.Attr | None = kwargs.get(_AXES_NAME)

        # axis may be provided as input to ReduceMean and as attribute to RMSNormalization
        if axis_attr and not get_constant_singleton_value(axis_attr):
            return match_result.fail(
                f"Axes input must be a single element constant, got {axis_attr}"
            )
        if (
            axis_attr
            and axes_attr
            and get_constant_singleton_value(axis_attr)
            != get_constant_singleton_value(axes_attr)
        ):
            return match_result.fail(
                f"Axis value mismatch between reduction ops, got: {axis_attr} vs {axes_attr}"
            )
        if axes_attr and not len(axes_attr.as_ints()) == 1:
            return match_result.fail(
                f"Only single axis LayerNormalization is supported, got axes={axes_attr.value}"
            )

        # Espilon may be tensor input or attribute of RMSNormalization subgraph
        eps_value = get_constant_singleton_value(epsilon)
        if eps_value is not None and eps_value < 0:
            return match_result.fail(
                f"Epsilon must be a positive constant, got {eps_value}"
            )

        return match_result

    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        input_x: onnx_ir.Value,
        scale: onnx_ir.Value,
        bias: onnx_ir.Value,
        **kwargs,
    ) -> onnx_ir.Value:
        """
        Defines the fused replacement for the matched decomposed pattern.
        """
        axes = kwargs.get(_AXES_NAME) or kwargs.get(_AXIS_NAME)
        epsilon = kwargs.get(_EPS_NAME)

        result = op.LayerNormalization(
            input_x,
            scale,
            bias,
            axis=get_constant_singleton_value(axes),
            epsilon=get_constant_singleton_value(epsilon),
            _domain=AIMET_SUPERGROUP_DOMAIN,
        )

        return result
