# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX supergroup fusion implementation"""

import onnx_ir
from onnxscript.rewriter import pattern
from .fusion_registry import FUSION_PASS_REGISTRY


def fuse_supergroups(
    model: onnx_ir.Model, patterns: list[str], *, verbose: int | None = None
) -> onnx_ir.Model:
    """
    Fuses specified patterns in the ONNX model into supergroups.

    This function identifies fusable decomposed operators (e.g., decomposed LayerNormalization)
    in the ONNX model and replaces them with onnx_ir.Functions representing the fused operator.

    Args:
        model: The ONNX model to process.
        patterns: List of pattern names to fuse (available patterns: {"LayerNormalization"}).

    Returns:
        onnx_ir.Model: The modified model with each matched pattern represented as an onnx_ir Function.

    Example:
        >>> model = onnx_ir.load("decomposed_model.onnx")
        >>> fused = fuse_supergroups(model, patterns=['LayerNormalization'])
    """
    unknown_patterns = [p for p in patterns if p not in FUSION_PASS_REGISTRY]
    if unknown_patterns:
        raise ValueError(
            f"Unknown pattern names: {unknown_patterns}. "
            f"Available patterns: {list(FUSION_PASS_REGISTRY.passes.keys())}"
        )

    if not patterns:
        return model

    # Create rewrite rule set to match specified patterns into onnx functions
    fusion_rules = [
        pattern for name in patterns for pattern in FUSION_PASS_REGISTRY[name]
    ]
    rule_set = pattern.RewriteRuleSet(fusion_rules)

    # Apply the rewrite rules to the model
    count = rule_set.apply_to_model(model, verbose=verbose)
    if count:
        onnx_ir.passes.common.RemoveUnusedNodesPass().call(model)

    return model
