# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Decoder linear-op role mapping.

Given decoder block boundaries and the active norms, classifies every weighted
linear op into its role (qkv / o_proj / gate_up / down_proj) per block, plus the
model-level embed_tokens and lm_head. Technique-agnostic: it describes the
structure of a decoder stack without knowing about any specific quantization
technique.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.block_topology.block_boundaries import (
    tensor_to_first_consumer_index,
)
from aimet_onnx.experimental.block_topology.norm_detection import (
    ActiveNorm,
    find_active_norms,
)
from aimet_onnx.experimental.block_topology.weight_utils import (
    get_weight_product,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BlockTopology)

_LINEAR_TYPES = frozenset(("MatMul", "Gemm", "Conv"))
_EMBEDDING_TYPES = frozenset(("Gather",))


def _collect_past_key_input_names_in_order(model: ModelProto) -> List[str]:
    """Return ``past_key_*`` graph input names in declaration order.

    HF/optimum LLM exports with a KV-cache expose one such input per decoder
    block. Prefill-only exports have none.
    """
    return [
        inp.name
        for inp in model.graph.input
        if "past_key" in inp.name or "past_k_" in inp.name
    ]


def _is_embedding_table_gather(op: Op) -> bool:
    """Return True if ``op`` is a token-embedding ``Gather`` (data is a 2-D table).

    A real embedding ``Gather`` has the embedding *table* as its first (data)
    input — a static rank-2 ``[vocab, hidden]`` initializer. Other Gathers in
    the prologue (e.g. position-id lookups, ``shape``-derived indexers) hold
    static scalar or 1-D constants on input 0 and must be excluded.

    :param op: Candidate Gather op.
    :return: True iff ``op`` looks like a token-embedding lookup.
    """
    if not op.inputs:
        return False
    data_inp = op.inputs[0]
    if not (data_inp.is_parm or data_inp.is_const):
        return False
    shape = getattr(data_inp, "shape", None)
    if shape is None:
        return False
    return len(shape) >= 2


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
    :param past_key_input_names: Raw ``past_key_*`` graph inputs in declaration
        order, collected tolerantly (empty for prefill-only exports without a
        KV-cache). Pairing these to ``blocks`` and validating that their count
        matches the block count are R3's responsibility, not this builder's —
        R1-only and prefill-only flows do not require KV-cache inputs.
    """

    embed_tokens: List[Op] = field(default_factory=list)
    lm_head: List[Op] = field(default_factory=list)
    blocks: List[DecoderBlockRoleMap] = field(default_factory=list)
    past_key_input_names: List[str] = field(default_factory=list)


def get_decoder_role_map(
    connected_graph: ConnectedGraph,
    block_boundaries: List[Tuple[str, str]],
    active_norms: Optional[List[ActiveNorm]] = None,
    active_norms_per_block: int = 2,
) -> DecoderModelRoleMap:
    """Build the decoder linear role map from pre-computed block boundaries.

    Per-block roles. Reads are the weighted linears downstream of each active
    norm; writes are found by walking back from the residual output tensor to
    the weighted linear(s) that feed it:

    * ``qkv_linears``     — ``downstream_linears`` of the first active norm in
      the block (input norm).
    * ``o_proj``          — weighted linear(s) that write the attention residual
      output (the post-attention norm input).
    * ``gate_up_linears`` — ``downstream_linears`` of the second active norm in
      the block (post-attention norm).
    * ``down_proj``       — weighted linear(s) that write the block residual
      output (the block-end tensor).

    Model-level roles:

    * ``lm_head``         — downstream linears of active norms at or after the
      last block boundary (outside all decoder blocks).
    * ``embed_tokens``    — Gather ops with a static-weight input that appear
      before the first block boundary.

    :param connected_graph: ConnectedGraph built from the model.
    :param block_boundaries: List of ``(start_tensor, end_tensor)`` residual-stream
        tensor names, as returned by :func:`get_decoder_block_boundaries`.
    :param active_norms: Active norms in topological order. Recomputed via
        :func:`find_active_norms` when not supplied; pass a precomputed value to
        avoid a redundant graph scan.
    :param active_norms_per_block: Expected number of active norms per decoder
        block. Must match the value used in :func:`get_decoder_block_boundaries`.
        Defaults to 2 (Llama/Qwen2/Mistral/Phi family).
    :return: DecoderModelRoleMap with all roles populated.
    """
    op_topo_idx = {op: i for i, op in enumerate(connected_graph.ordered_ops)}
    if active_norms is None:
        active_norms = find_active_norms(connected_graph.model, connected_graph)
    boundary_topo = tensor_to_first_consumer_index(connected_graph)

    # tensor name -> producing Op
    producer_by_tensor = {
        out.name: op for op in connected_graph.ordered_ops for out in op.outputs
    }

    result = DecoderModelRoleMap()

    for block_idx, (start_tensor, end_tensor) in enumerate(block_boundaries):
        start_topo = boundary_topo[start_tensor]
        end_topo = boundary_topo[end_tensor]

        # Active norms whose norm_op falls in [start_topo, end_topo).
        # index 0 = input_norm (pre-attention),
        # index 1 = post_attn_norm (pre-MLP).
        block_active_norms = [
            active_norm
            for active_norm in active_norms
            if start_topo <= op_topo_idx[active_norm.norm_op] < end_topo
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

        qkv_linears = list(input_norm.downstream_linears)
        gate_up_linears = list(post_attn_norm.downstream_linears)

        intermediate_tensor = post_attn_norm.norm_op.inputs[0].name
        o_proj_candidates = _find_nearest_upstream_linears(
            intermediate_tensor, start_tensor, producer_by_tensor, op_topo_idx
        )
        if not o_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no attention residual writer (o_proj) found "
                f"for residual output '{intermediate_tensor}'."
            )

        down_proj_candidates = _find_nearest_upstream_linears(
            end_tensor, intermediate_tensor, producer_by_tensor, op_topo_idx
        )
        if not down_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no MLP residual writer (down_proj) found "
                f"for residual output '{end_tensor}'."
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

    last_end_topo = boundary_topo[block_boundaries[-1][1]]
    result.lm_head = [
        op
        for an in active_norms
        if op_topo_idx[an.norm_op] >= last_end_topo
        for op in an.downstream_linears
    ]
    if not result.lm_head:
        _logger.debug(
            "lm_head not detected: no active norm found after the last block boundary."
        )
    else:
        _logger.debug("lm_head: %s", [op.name for op in result.lm_head])

    first_start_topo = boundary_topo[block_boundaries[0][0]]
    result.embed_tokens = [
        op
        for op in connected_graph.ordered_ops
        if op.type in _EMBEDDING_TYPES
        and op_topo_idx[op] < first_start_topo
        and _is_embedding_table_gather(op)
    ]
    if not result.embed_tokens:
        _logger.info(
            "Backbone: embed_tokens not detected, no Gather op with a static weight found before "
            "the first block boundary. This is expected for VLM backbones exported with "
            "use_inputs_embeds=True. Rotate embedding.pth separately."
        )
    _logger.debug("embed_tokens: %s", [op.name for op in result.embed_tokens])

    # Collected tolerantly: prefill-only / R1-only flows leave this empty and
    # never require KV-cache inputs. R3 validates the count against blocks.
    result.past_key_input_names = _collect_past_key_input_names_in_order(
        connected_graph.model
    )
    _logger.debug("past_key inputs: %s", result.past_key_input_names)

    _logger.info(
        "Backbone: %d block(s), embed_tokens=%s, lm_head=%s.",
        len(result.blocks),
        [op.name for op in result.embed_tokens],
        [op.name for op in result.lm_head],
    )

    return result


def _find_nearest_upstream_linears(
    target_tensor: str,
    boundary_tensor: str,
    producer_by_tensor: dict,
    op_topo_idx: dict,
) -> List[Op]:
    """Nearest weighted linears feeding target_tensor, via a backward walk.

    Walks backward from the op producing target_tensor and collects the first
    weighted linear (MatMul/Gemm/Conv with a static weight) on each path,
    stopping there; any other op type is crossed transparently.

    NOTE: The walk is fenced at boundary_tensor's producer: that op and anything
    earlier are skipped, so the walk does not cross into the previous block.

    :param target_tensor: Tensor whose upstream linears are wanted (walk start).
    :param boundary_tensor: Upstream edge; its producer and earlier ops are the
        lower fence.
    :param producer_by_tensor: Map of tensor name -> producing Op.
    :param op_topo_idx: Map of Op -> topological index.
    :return: The nearest weighted linear op on each backward path.
    """
    start = producer_by_tensor.get(target_tensor)
    if start is None:
        return []

    fence_op = producer_by_tensor.get(boundary_tensor)
    lo = op_topo_idx[fence_op] if fence_op is not None else -1

    linears = []
    seen = set()
    queue = [start]
    while queue:
        op = queue.pop()
        if op in seen:
            continue
        seen.add(op)
        if op_topo_idx.get(op, -1) <= lo:
            continue
        if op.type in _LINEAR_TYPES and get_weight_product(op)[0] is not None:
            linears.append(op)
            continue  # this linear shadows everything upstream of it
        queue.extend(op.input_ops)
    return linears
