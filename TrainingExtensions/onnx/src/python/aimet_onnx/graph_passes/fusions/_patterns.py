# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Common patterns for pattern matching"""

from onnxscript.rewriter import pattern
from .fusion_registry import AIMET_SUPERGROUP_DOMAIN


def square(op: pattern.OpsetPatternBuilder, tensor: pattern.Var):
    """Matches `x * x` or `Pow(x, 2)`"""
    exp = pattern.OrValue([pattern.Constant(2.0), pattern.Constant([2.0])])
    return pattern.OrValue([tensor * tensor, op.Pow(tensor, exp)])


def reciprocal(op: pattern.OpsetPatternBuilder, x: pattern.Var):
    """Matches `Reciprocal(x)` or `1 / x`"""
    return pattern.OrValue([op.Reciprocal(x), pattern.Constant(1.0) / x])


def add(op: pattern.OpsetPatternBuilder, x: pattern.Var, y: pattern.Var):
    """Commutative add pattern"""
    return pattern.OrValue([op.Add(x, y), op.Add(y, x)])


def mul(op: pattern.OpsetPatternBuilder, x: pattern.Var, y: pattern.Var):
    """Commutative mul pattern"""
    return pattern.OrValue([op.Mul(x, y), op.Mul(y, x)])


def div(
    op: pattern.OpsetPatternBuilder, numerator: pattern.Var, denominator: pattern.Var
):
    """Matches `numerator / denominator` or `numerator * Reciprocal(denominator)`"""
    inv_denom = reciprocal(op, denominator)
    return pattern.OrValue([numerator / denominator, mul(op, numerator, inv_denom)])


def reduce_mean(op: pattern.OpsetPatternBuilder, x: pattern.Var, axes_name: str):
    """Matches ReduceMean(x) for opsets {13, 18}"""
    # Note: In opset>=18, axes is an optional input to ReduceMean rather than attribute
    axes_var = pattern.Var(axes_name)
    axes_attr = pattern.AttrVar(axes_name, can_match_none=True)
    return pattern.OrValue(
        [op.ReduceMean(x, axes_var), op.ReduceMean(x, axes=axes_attr)]
    )


def non_affine_rms_normalize(
    op: pattern.OpsetPatternBuilder,
    x: pattern.Var,
    epsilon: str,
    axes: str,
    pattern_idx: int | None = None,
):
    """
    Returns non-affine RMSNormalization pattern
                x
            +---+---+
            |       |
        Pow(x, 2)   |
            |       |
        ReduceMean  |
            |       |
            Add     |
            |       |
            Sqrt    |
            |       |
            +---+---+
                Mul

    Args:
        op: OpsetPatternBuilder
        x: Input to be normalized
        epsilon: Epsilon used in Add operation
        axes: Handle to use for axes attribute. This can be matched to Var or AttrVar depending on opset version
        pattern_idx: If specified, returns a concrete pattern node rather than output of OrValue operation. Due to limitation
         in onnxscript, this is required when the output is returned directly as an output of RewriteRuleClassBase.pattern method.
         Valid values are {None, 0, 1, 2}.
    """
    if pattern_idx not in {None, 0, 1, 2}:
        raise ValueError(f"Invalid value passed for pattern_idx: {pattern_idx}")
    squared = square(op, x)
    mean = reduce_mean(op, squared, axes)
    mean_eps = add(op, mean, pattern.Var(epsilon))
    sqrt_mean = op.Sqrt(mean_eps)
    inv_sqrt_mean = reciprocal(op, sqrt_mean)
    normalize_patterns = [
        x / sqrt_mean,
        x * inv_sqrt_mean,
        inv_sqrt_mean * x,
    ]
    return (
        pattern.OrValue(normalize_patterns)
        if pattern_idx is None
        else normalize_patterns[pattern_idx]
    )


def rms_normalize(
    op: pattern.OpsetPatternBuilder,
    x: pattern.Var,
    scale: pattern.Var,
    epsilon: str,
    axes: str,
    output_type: str,
    pattern_idx: int | None = None,
):
    """
    Matches affine RMSNormalization pattern with optional internal cast (does not include input cast)

    Args:
        op: OpsetPatternBuilder
        x: Input to normalize
        scale: affine scaling weight
        epsilon: Handle for epsilon value. Can be matched to pattern.Var or pattern.AttrVar
        axes: Handle for axes value. Can be matched to pattern.Var or pattern.AttrVar
        output_type: Handle for the optional trailing Cast's ``to`` attribute.
        pattern_idx: If specified returns a concrete pattern node rather than OrValue output. Valid values are {None, 0, 1}.
    """

    epsilon_attr = pattern.AttrVar(epsilon)
    axes_attr = pattern.AttrVar(axes, can_match_none=True)
    output_type_attr = pattern.AttrVar(output_type)

    normalized = pattern.OrValue(
        [
            non_affine_rms_normalize(op, x, epsilon, axes),
            op.RMSNormalization(
                x, epsilon=epsilon_attr, axis=axes_attr, _domain=AIMET_SUPERGROUP_DOMAIN
            ),
        ]
    )

    normalized = pattern.OrValue([normalized, op.Cast(normalized, to=output_type_attr)])
    output_patterns = [
        normalized * scale,
        scale * normalized,
    ]
    return (
        pattern.OrValue(output_patterns)
        if pattern_idx is None
        else output_patterns[pattern_idx]
    )
