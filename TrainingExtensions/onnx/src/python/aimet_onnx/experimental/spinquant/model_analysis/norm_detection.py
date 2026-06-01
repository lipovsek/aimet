# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Affine RMSNorm detection for SpinQuant.

An *active norm* is an affine RMSNorm whose gamma-scale Mul has at least one
downstream weight MatMul/Gemm/Conv reachable through grid-preserving ops.
Internal norms (e.g. Qwen3 q_norm/k_norm) whose outputs feed into attention ops
before reaching any linear weight are excluded automatically.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.graph_passes.passes.common_patterns import match_rms_norm_pattern
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.spinquant.model_analysis.weight_utils import (
    get_weight_product,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


# Op types that only reshape/reformat activations without changing the
# mathematical relationship between gamma and the downstream linear weight.
_OP_OUTPUTS_TO_IGNORE = [
    "Unsqueeze",
    "Squeeze",
    "Transpose",
    "Reshape",
    "Flatten",
    "Cast",
]


@dataclass
class ActiveNorm:
    """An affine RMSNorm that has at least one downstream weight linear op.

    :param norm_op: The Pow/Mul starting op of the matched RMSNorm pattern.
    :param scale_name: Name of the gamma (scale) initializer in the model.
    :param downstream_linears: MatMul/Gemm/Conv ops reachable from the scale Mul.
    """

    norm_op: Op
    scale_name: str
    downstream_linears: List[Op] = field(default_factory=list)


def find_active_norms(
    model: ModelProto, connected_graph: ConnectedGraph
) -> List[ActiveNorm]:
    """Return all affine RMSNorms with at least one downstream weight linear op.

    Iterates ``connected_graph.ordered_ops`` in topological order and collects
    every op that starts an affine RMSNorm pattern whose gamma-scale Mul has at
    least one downstream weight MatMul/Gemm/Conv (reachable through reshape-only
    ops). Norms with no weight consumers are omitted.

    :param model: ONNX ModelProto.
    :param connected_graph: ConnectedGraph built from ``model``.
    :return: ``ActiveNorm`` objects in topological order.
    """
    result = []
    for op in connected_graph.ordered_ops:
        match = _find_norm_scale_and_consumers(op, model)
        if match is None:
            continue
        scale_name, downstream_linears = match
        if not downstream_linears:
            _logger.debug(
                "RMSNorm scale '%s' (op '%s'): no downstream weight linears, skipping.",
                scale_name,
                op.name,
            )
            continue
        result.append(
            ActiveNorm(
                norm_op=op, scale_name=scale_name, downstream_linears=downstream_linears
            )
        )
    _logger.debug("Found %d active norm(s).", len(result))
    return result


def find_post_writing_norms(model: ModelProto, role_map) -> List[str]:
    """Return op names of affine RMSNorms immediately after writing layers.

    Used by R1 architecture compatibility checks: R1 absorption requires writing
    layers (o_proj, down_proj) to feed directly into the residual add, with no
    affine RMSNorm in between.

    :param model: ONNX ModelProto.
    :param role_map: DecoderModelRoleMap.
    :return: List of norm op names for detected post-writing norms.
    """
    found = []
    for block in role_map.blocks:
        for writing_op in block.o_proj + block.down_proj:
            for out_op in writing_op.output_ops:
                candidates = out_op.output_ops if out_op.type == "Cast" else [out_op]
                for candidate in candidates:
                    match = _find_norm_scale_and_consumers(candidate, model)
                    if match is not None:
                        found.append(candidate.name)
                        break
    return found


def _iter_linear_consumers(scale_mul_op: Op) -> List[Op]:
    """Return all downstream weight MatMul/Gemm/Conv ops reachable from ``scale_mul_op``.

    MatMul/Gemm/Conv ops can be direct consumers, or reached through a chain of
    reshape/reformat ops (Unsqueeze, Transpose, Cast, Reshape, ...) that adjust
    the activation layout into Conv-compatible format.

    :param scale_mul_op: The Mul op that applies the RMSNorm scale (gamma) to the
                         normalized activations. Its outputs are traversed to collect
                         downstream weight ops.
    :return: List of MatMul, Gemm, or Conv ops that consume the scaled activations
             and have a static weight initializer available for fusion.
    """
    result = []
    visited = set()
    queue = list(scale_mul_op.output_ops)
    while queue:
        consumer = queue.pop()
        if id(consumer) in visited:
            continue
        visited.add(id(consumer))
        if (
            consumer.type in ("MatMul", "Gemm")
            and get_weight_product(consumer)[0] is not None
        ):
            result.append(consumer)
        elif consumer.type in _OP_OUTPUTS_TO_IGNORE:
            queue.extend(consumer.output_ops)
        elif consumer.type == "Conv" and get_weight_product(consumer)[0] is not None:
            result.append(consumer)
    return result


def _find_norm_scale_and_consumers(
    op: Op, model: ModelProto
) -> Optional[Tuple[str, List[Op]]]:
    """If ``op`` starts an affine RMSNorm, return ``(scale_name, downstream_linears)``.

    Returns None if op does not start an affine RMSNorm (non-affine norms have
    no scale to fuse).

    :param op: The candidate starting op to check for an affine RMSNorm pattern.
    :param model: ONNX ModelProto used to look up initializer names when matching
                  the RMSNorm pattern.
    :return: A tuple (scale_initializer_name, downstream_linear_ops) if op starts
             an affine RMSNorm, where scale_initializer_name is the name of the gamma
             initializer and downstream_linear_ops is the list of weight MatMul/Gemm/Conv
             ops that follow the norm's scale multiply. Returns None if the op does not
             match an affine RMSNorm pattern.
    """
    norm_ops = match_rms_norm_pattern(op, model)
    if not norm_ops:
        return None

    # Fused RMSNormalization op: gamma is the second input by ONNX spec.
    # AIMET's supergroup fuser produces a non-affine call (only 1 input — X);
    # in that case gamma lives outside as a downstream Mul, so fall through.
    if len(norm_ops) == 1 and norm_ops[0].type == "RMSNormalization":
        rms_op = norm_ops[0]
        if len(rms_op.inputs) >= 2 and rms_op.parameters:
            downstream_linears = _iter_linear_consumers(rms_op)
            return rms_op.inputs[1].name, downstream_linears

    # match_rms_norm_pattern stops at the last Mul op. Check for the trailing
    # scale Mul: RMSNorm(x) * gamma, with an optional Cast in between.
    last_op = norm_ops[-1]
    if len(last_op.output_ops) != 1:
        return None  # non-affine norm, nothing to fuse
    next_op = last_op.output_ops[0]
    if next_op.type == "Cast" and len(next_op.output_ops) == 1:
        next_op = next_op.output_ops[0]
    if next_op.type != "Mul":
        return None  # non-affine norm, nothing to fuse

    scale_mul_op = next_op
    scale_inp = next((inp for inp in scale_mul_op.inputs if inp.is_const), None)
    if scale_inp is None:
        return None

    downstream_linears = _iter_linear_consumers(scale_mul_op)
    return scale_inp.name, downstream_linears
