# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LayerNormalization fusion pass for ONNX models"""

import onnx_ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from .ir_utils import get_constant_singleton_value
from .fusion_registry import register_fusion, AIMET_SUPERGROUP_DOMAIN


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
        epsilon: pattern.Var,
        scale: pattern.Var,
        bias: pattern.Var,
    ):
        """
        Defines the decomposed LayerNormalization pattern to match.
        """
        axes_var = pattern.AttrVar("axes")

        # E[x]
        mean = op.ReduceMean(input_x, axes=axes_var)

        # x - E[x]
        centered = input_x - mean

        # (x - E[x])^2
        pow_val = pattern.Constant(2.0)
        squared = pattern.OrValue([op.Pow(centered, pow_val), centered * centered])

        # Var[x] = E[(x - E[x])^2]
        variance = op.ReduceMean(squared, axes=axes_var) + epsilon

        # sqrt(Var[x] + epsilon)
        denominator = op.Sqrt(variance)

        # (x - E[x]) / sqrt(Var[x] + epsilon)
        normalized = centered / denominator

        # normalized * scale + bias
        return normalized * scale + bias

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        input_x: onnx_ir.Value,
        epsilon: onnx_ir.Value,
        scale: onnx_ir.Value,
        bias: onnx_ir.Value,
        **kwargs,
    ) -> rewriter.pattern.MatchResult:
        """
        Validates that a matched pattern satisfies additional constraints on LayerNormalization ops.
        """
        match_result = pattern.MatchResult()

        eps_value = get_constant_singleton_value(epsilon)
        if eps_value is None or eps_value <= 0:
            return match_result.fail(
                f"Epsilon must be a positive constant, got {eps_value}"
            )

        axes = kwargs.get("axes")
        if not isinstance(axes, onnx_ir.Attr) or not isinstance(
            axes.value, (list, tuple)
        ):
            return match_result.fail("Axes attribute is required for ReduceMean")

        if len(axes.value) != 1:
            return match_result.fail(
                f"Only single axis LayerNormalization is supported, got axes={axes}"
            )

        return match_result

    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        input_x: onnx_ir.Value,
        epsilon: onnx_ir.Value,
        scale: onnx_ir.Value,
        bias: onnx_ir.Value,
        **kwargs,
    ) -> onnx_ir.Value:
        """
        Defines the fused replacement for the matched decomposed pattern.
        """
        axes = kwargs.get("axes")
        epsilon_value = get_constant_singleton_value(epsilon)
        if epsilon_value is None:
            raise ValueError("Epsilon must be a constant singleton value")

        # Determine axis attribute
        if (
            not isinstance(axes, onnx_ir.Attr)
            or not isinstance(axes.value, (list, tuple))
            or len(axes.value) != 1
        ):
            raise ValueError(
                f"Single integer axis attribute is required, got axes={axes}"
            )

        axis_attr = int(axes.value[0])

        result = op.LayerNormalization(
            input_x,
            scale,
            bias,
            axis=axis_attr,
            epsilon=epsilon_value,
            _domain=AIMET_SUPERGROUP_DOMAIN,
        )

        return result
