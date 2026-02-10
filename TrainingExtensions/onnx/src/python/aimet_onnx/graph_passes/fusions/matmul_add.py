# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""MatmulAdd fusion pass for onnx-ir models"""

import onnx_ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from .ir_utils import get_constant_as_array
from .fusion_registry import register_fusion, AIMET_SUPERGROUP_DOMAIN


@register_fusion("MatmulAdd", bias_first=False, trans_b=False)
@register_fusion("MatmulAdd", bias_first=False, trans_b=True)
@register_fusion("MatmulAdd", bias_first=True, trans_b=False)
@register_fusion("MatmulAdd", bias_first=True, trans_b=True)
class MatmulAddFusion(pattern.RewriteRuleClassBase):
    """
    Fuses decomposed torch.nn.Linear pattern (Matmul -> Add) into a Gemm node.

    Implements pattern matching and replacement for the decomposed Linear operation:

    Linear(x, W, b) = x @ W^T + b

    Expected decomposed graph patterns:
        MatMul(x, W) + b -> Gemm(x, W, b)
        MatMul(x, Transpose(W_t)) + b -> Gemm(x, W, b, transB=True)
    """

    def __init__(
        self, *args, bias_first: bool = False, trans_b: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bias_first = bias_first
        self.trans_b = trans_b

    # pylint: disable=arguments-differ
    def pattern(
        self,
        op: pattern.OpsetPatternBuilder,
        input_x: pattern.Var,
        weight: pattern.Var,
        bias: pattern.Var,
    ):
        """
        Defines the decomposed Matmul->Add pattern to match.
        """
        if self.trans_b:
            # Match both with and without explicit "perm" attribute
            weight = op.Transpose(
                weight, perm=pattern.AttrVar("perm", can_match_none=True)
            )

        matmul_output = op.MatMul(input_x, weight)

        if self.bias_first:
            return bias + matmul_output

        return matmul_output + bias

    # pylint: disable=unused-argument
    def check(
        self,
        context: rewriter.MatchContext,
        input_x: onnx_ir.Value,
        weight: onnx_ir.Value,
        bias: onnx_ir.Value,
        **kwargs,
    ) -> rewriter.pattern.MatchResult:
        """
        Validates that a matched pattern satisfies additional constraints.
        """
        match_result = pattern.MatchResult()

        weight_array = get_constant_as_array(weight)
        if weight_array is None:
            return match_result.fail("Weight must be a constant tensor")

        if len(weight_array.shape) != 2:
            return match_result.fail(
                f"Weight must be rank 2, got shape {weight_array.shape}"
            )

        bias_array = get_constant_as_array(bias)
        if bias_array is None:
            return match_result.fail("Bias must be a constant tensor")

        if len(bias_array.shape) != 1:
            return match_result.fail(
                f"Bias must be rank 1, got shape {bias_array.shape}"
            )

        # Validate perm attribute if Transpose was matched
        perm = kwargs.get("perm")
        if perm is not None:
            perm_value = perm.value
            # For 2D transpose, perm should be [1, 0]
            if list(perm_value) != [1, 0]:
                return match_result.fail(
                    f"Only 2D transpose with perm=[1, 0] is supported, got perm={perm_value}"
                )

        return match_result

    def rewrite(
        self,
        op: onnx_ir._tape.Builder,
        input_x: onnx_ir.Value,
        weight: onnx_ir.Value,
        bias: onnx_ir.Value,
        **kwargs,
    ) -> onnx_ir.Value:
        """
        Creates a Gemm node with the appropriate transB attribute based on
        whether the pattern included a Transpose node.
        """
        result = op.Gemm(
            input_x,
            weight,
            bias,
            transB=self.trans_b,
            _domain=AIMET_SUPERGROUP_DOMAIN,
        )

        return result
