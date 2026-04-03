# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Topology-driven decoder block boundary detection.

An *active norm* is an affine RMSNorm whose gamma-scale Mul has at least one
downstream weight MatMul/Gemm/Conv reachable through grid-preserving ops. Internal
norms (e.g. Qwen3 q_norm/k_norm) whose outputs feed into attention ops before
reaching any linear weight are excluded automatically.

Decoder block detection relies on the premise that transformer decoder stacks
contain exactly ``k`` active norms per block plus one final active norm:

    active norms (topological order): [n0, n1, ..., n_{kN}]
    block i boundaries               : (n_{k*i}, n_{k*(i+1)})

Most architectures use k=2 (pre-attention norm + pre-FFN norm, e.g. Llama/Qwen).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from aimet_onnx.common.utils import AimetLogger

from aimet_onnx.experimental.spinquant.fuse_norm import (
    ActiveNorm,
    find_active_norms,
    _get_weight_product,
)
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)

_LINEAR_TYPES = frozenset(("MatMul", "Gemm", "Conv"))
_EMBEDDING_TYPES = frozenset(("Gather",))


@dataclass
class DecoderBlockRoleMap:
    """Role map for the weighted linear ops in one decoder block.

    :param qkv_linears: Q/K/V projection ops that read from the residual
        stream through the block's input norm.
    :param o_proj: Attention-output projection op(s) that write back to the
        residual.
    :param gate_up_linears: Gate and up projection ops that read from the
        residual stream through the block's post-attention norm.
    :param down_proj: MLP-output projection op(s) that write back to the
        residual.
    """

    qkv_linears: List[Op] = field(default_factory=list)
    o_proj: List[Op] = field(default_factory=list)
    gate_up_linears: List[Op] = field(default_factory=list)
    down_proj: List[Op] = field(default_factory=list)


@dataclass
class DecoderModelRoleMap:
    """Linear layer role map for an ONNX decoder-stack model.

    :param embed_tokens: Token-embedding Gather op(s) that produce the initial
        residual-stream activations.
    :param lm_head: Vocabulary-projection linear(s) downstream of the final norm.
    :param blocks: Per-decoder-block role maps in topological order.
    """

    embed_tokens: List[Op] = field(default_factory=list)
    lm_head: List[Op] = field(default_factory=list)
    blocks: List[DecoderBlockRoleMap] = field(default_factory=list)


def get_decoder_block_boundaries(
    model: ModelProto,
    connected_graph: ConnectedGraph,
    expected_num_blocks: Optional[int] = None,
    active_norms_per_block: Optional[int] = None,
) -> Tuple[List[Tuple[Op, Op]], List[ActiveNorm]]:
    """Return block boundaries and active norms for each decoder block.

    The structural assumption is ``total_active_norms == k * N + 1``, where ``k`` is the
    number of active norms per block and ``N`` is the number of decoder blocks.

    Block *i* spans from ``active[k*i].norm_op`` to ``active[k*(i+1)].norm_op``.

    :param model: ONNX ModelProto.
    :param connected_graph: ConnectedGraph built from model.
    :param expected_num_blocks: If provided, raises ``ValueError`` when the
        detected block count does not match.
    :param active_norms_per_block: Number of **active** norms per decoder block
      (norms whose gamma-scale Mul has at least one downstream weight linear).
      Defaults to 2 (Llama/Qwen2/Mistral/Phi family).
      NOTE: Do NOT count internal norms (e.g. Qwen3 q_norm/k_norm) — these
      are filtered out automatically and must not be included in this count.
    :return: Tuple of (block_boundaries, active_norms), where block_boundaries is a list
        of ``(start_op, end_op)`` tuples one per decoder block, and active_norms is the
        full list of active norms in topological order. Returning active_norms avoids
        rescanning the graph in downstream steps (role mapping, norm fusion).
    :raises ValueError: If active norm count is inconsistent with ``k``, or if
        ``expected_num_blocks`` is given and does not match the detected count.
    """
    active_norms = find_active_norms(model, connected_graph)
    num_active_norms = len(active_norms)

    if num_active_norms == 0:
        raise ValueError(
            "No active RMSNorms found. The model may use a normalization pattern "
            "not covered by match_rms_norm_pattern, or all norms lack downstream "
            "weight linear layers."
        )

    resolved_norms_per_block: int
    if active_norms_per_block is not None:
        resolved_norms_per_block = active_norms_per_block
    elif expected_num_blocks is not None:
        remainder = num_active_norms - 1
        if remainder <= 0 or remainder % expected_num_blocks != 0:
            raise ValueError(
                f"Cannot infer active_norms_per_block: {num_active_norms} active norm(s) and "
                f"expected_num_blocks={expected_num_blocks} are inconsistent "
                f"(require (num_active_norms-1) divisible by expected_num_blocks)."
            )
        resolved_norms_per_block = remainder // expected_num_blocks
    else:
        resolved_norms_per_block = 2  # default: Llama/Qwen2/Mistral/Phi family
        _logger.warning(
            "Neither expected_num_blocks nor active_norms_per_block was provided. "
            "Defaulting to active_norms_per_block=2 (Llama/Qwen2/Mistral/Phi). "
            "Pass expected_num_blocks=<N> to validate the detected block count."
        )

    if (num_active_norms - 1) % resolved_norms_per_block != 0:
        raise ValueError(
            f"Active norm count {num_active_norms} is inconsistent with active_norms_per_block={resolved_norms_per_block}: "
            f"expected (num_active_norms-1) to be divisible by {resolved_norms_per_block} "
            f"(i.e. resolved_norms_per_block*N+1 active norms for N decoder blocks). "
        )

    num_blocks = (num_active_norms - 1) // resolved_norms_per_block
    if expected_num_blocks is not None and num_blocks != expected_num_blocks:
        raise ValueError(
            f"Expected {expected_num_blocks} decoder blocks but detected {num_blocks}."
        )
    _logger.info(
        "Detected %d decoder block(s) from %d active norm(s) (%d per block).",
        num_blocks,
        num_active_norms,
        resolved_norms_per_block,
    )
    block_boundaries = [
        (
            active_norms[resolved_norms_per_block * i].norm_op,
            active_norms[resolved_norms_per_block * (i + 1)].norm_op,
        )
        for i in range(num_blocks)
    ]

    return block_boundaries, active_norms


def get_decoder_role_map(
    connected_graph: ConnectedGraph,
    block_boundaries: List[Tuple[Op, Op]],
    active_norms: List[ActiveNorm],
    active_norms_per_block: int = 2,
) -> DecoderModelRoleMap:
    """Build the decoder linear role map from pre-computed block detection results.

    Per-block roles are detected via topological-index range queries:

    * ``qkv_linears``     — ``downstream_linears`` of the first active norm in
      the block (input norm).
    * ``o_proj``          — all weighted linears strictly between the input norm
      and the post-attention norm, excluding QKV projections.
    * ``gate_up_linears`` — ``downstream_linears`` of the second active norm in
      the block (post-attention norm).
    * ``down_proj``       — all weighted linears strictly between the
      post-attention norm and the block end, excluding gate/up projections.

    Model-level roles:

    * ``lm_head``         — downstream linears of active norms at or after the
      last block boundary (outside all decoder blocks).
    * ``embed_tokens``    — Gather ops with a static-weight input that appear
      before the first block boundary.

    :param connected_graph: ConnectedGraph built from the model.
    :param block_boundaries: List of ``(start_op, end_op)`` tuples
    :param active_norms: Active norms in topological order
    :param active_norms_per_block: Expected number of active norms per decoder
        block. Must match the value used in :func:`get_decoder_block_boundaries`.
        Defaults to 2 (Llama/Qwen2/Mistral/Phi family).
    :return: :class:`DecoderModelRoleMap` with all roles populated.
    :raises ValueError: If the active norm count inside any block does not equal
        ``active_norms_per_block``; if no ``o_proj`` / ``down_proj`` candidates
        are found in any block; if ``lm_head`` or ``embed_tokens`` are not
        detected.
    """
    op_topo_idx = {id(op): i for i, op in enumerate(connected_graph.ordered_ops)}

    weighted_linears_topo = [
        op
        for op in connected_graph.ordered_ops
        if op.type in _LINEAR_TYPES and _get_weight_product(op)[0] is not None
    ]

    result = DecoderModelRoleMap()

    for block_idx, (start_op, end_op) in enumerate(block_boundaries):
        start_topo = op_topo_idx[id(start_op)]
        end_topo = op_topo_idx[id(end_op)]

        # Active norms whose norm_op falls in [start_topo, end_topo).
        # index 0 = input_norm (pre-attention),
        # index 1 = post_attn_norm (pre-MLP).
        block_active_norms = [
            active_norm
            for active_norm in active_norms
            if start_topo <= op_topo_idx[id(active_norm.norm_op)] < end_topo
        ]
        if len(block_active_norms) != active_norms_per_block:
            raise ValueError(
                f"Block {block_idx}: expected exactly {active_norms_per_block} active "
                f"norm(s) in topo range [{start_topo}, {end_topo}), "
                f"found {len(block_active_norms)}. "
                f"Ensure active_norms_per_block={active_norms_per_block} matches the "
                "value used in get_decoder_block_boundaries."
            )

        input_norm = block_active_norms[0]
        post_attn_norm = block_active_norms[1]
        attn_end_topo = op_topo_idx[id(post_attn_norm.norm_op)]

        qkv_linears = list(input_norm.downstream_linears)
        gate_up_linears = list(post_attn_norm.downstream_linears)
        qkv_ids = {id(op) for op in qkv_linears}
        gate_up_ids = {id(op) for op in gate_up_linears}

        # o_proj: all weighted linear op(s) strictly between input_norm and
        # post_attn_norm that are not QKV projections.
        o_proj_candidates = [
            op
            for op in weighted_linears_topo
            if start_topo < op_topo_idx[id(op)] < attn_end_topo
            and id(op) not in qkv_ids
        ]
        if not o_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no o_proj found in topo range "
                f"({start_topo}, {attn_end_topo})."
            )

        # down_proj: all weighted linear op(s) strictly between post_attn_norm and
        # end_op that are not gate/up projections.
        down_proj_candidates = [
            op
            for op in weighted_linears_topo
            if attn_end_topo < op_topo_idx[id(op)] < end_topo
            and id(op) not in gate_up_ids
        ]
        if not down_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no down_proj found in topo range "
                f"({attn_end_topo}, {end_topo})."
            )

        result.blocks.append(
            DecoderBlockRoleMap(
                qkv_linears=qkv_linears,
                o_proj=o_proj_candidates,
                gate_up_linears=gate_up_linears,
                down_proj=down_proj_candidates,
            )
        )
        _logger.debug(
            "Block %d: qkv=%s  o_proj=%s  gate_up=%s  down_proj=%s",
            block_idx,
            [op.name for op in qkv_linears],
            [op.name for op in o_proj_candidates],
            [op.name for op in gate_up_linears],
            [op.name for op in down_proj_candidates],
        )

    block_role_counts = [
        (
            len(b.qkv_linears),
            len(b.o_proj),
            len(b.gate_up_linears),
            len(b.down_proj),
        )
        for b in result.blocks
    ]
    if len(set(block_role_counts)) > 1:
        _logger.warning(
            "Inconsistent role shapes across %d decoder blocks — downstream algorithms "
            "may not apply correctly. Per-block shapes "
            "(n_qkv, n_o_proj, n_gate_up, n_down_proj): %s",
            len(result.blocks),
            block_role_counts,
        )

    last_end_topo = op_topo_idx[id(block_boundaries[-1][1])]
    result.lm_head = [
        op
        for an in active_norms
        if op_topo_idx[id(an.norm_op)] >= last_end_topo
        for op in an.downstream_linears
    ]
    if not result.lm_head:
        raise ValueError(
            "lm_head not detected: no active norm found after the last block boundary. "
            "The model must have a final norm with at least one downstream linear op "
            "outside all decoder block boundaries."
        )
    _logger.debug("lm_head: %s", [op.name for op in result.lm_head])

    first_start_topo = op_topo_idx[id(block_boundaries[0][0])]
    result.embed_tokens = [
        op
        for op in connected_graph.ordered_ops
        if op.type in _EMBEDDING_TYPES
        and op_topo_idx[id(op)] < first_start_topo
        and any(inp.is_parm or inp.is_const for inp in op.inputs)
    ]
    if not result.embed_tokens:
        raise ValueError(
            "embed_tokens not detected: no Gather op with a static weight found before "
            "the first block boundary. The model must have a token embedding op "
            "before the first decoder block."
        )
    _logger.debug("embed_tokens: %s", [op.name for op in result.embed_tokens])

    return result
