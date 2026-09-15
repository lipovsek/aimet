# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM decoder-stack topology.

Describes the structure of an ONNX decoder-stack model at two levels:

* *block level* — where each decoder block starts/ends on the residual stream
  (from :func:`get_decoder_block_boundaries`); and
* *intra-block level* — the individual weighted projections
  (q/k/v/o, gate/up/down, or their fused variants) and the two dynamic
  (non-weighted) attention MatMuls (Q·Kᵀ and softmax·V) inside each block,
  plus the model-level embed_tokens and lm_head.

Weighted read projections are grouped coarsely by the active norm they read
from (the ``qkv`` and ``gate_up`` :class:`LinearGroup`\\ s), and each group also
carries a fine-grained role split (``q_proj`` / ``k_proj`` / ``v_proj`` /
``gate_proj`` / ``up_proj``) derived from module names by :mod:`layer_roles`.
The dynamic MatMuls are found by pure graph topology.

Technique-agnostic: it describes a decoder stack without knowing about any
specific quantization technique. Everything here works on the analysis IR (see
:mod:`ir_analysis`) and reports results by ONNX name (see
:mod:`topology_types`), so a topology needs no graph object to stay meaningful;
techniques that must mutate raw ``NodeProto`` edges (e.g. SpinQuant R3) derive
their insertion anchors from this topology separately.

A consumer that goes on to *rewrite* the graph re-attaches an
:class:`onnx_ir.Model` with :func:`~.ir_adapter.resolve_topology`, which hands
back the same structure carrying ``onnx_ir.Node`` / ``onnx_ir.Value`` handles.
"""

from typing import Dict, List, Optional, Pattern, Tuple

import onnx_ir

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.ir_utils import static_tensor
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.llm_topology import ir_analysis
from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries_in_ir,
)
from aimet_onnx.experimental.llm_topology.layer_roles import (
    LinearRole,
)
from aimet_onnx.experimental.llm_topology.norm_detection import (
    ActiveNorm,
    find_active_norms_in_ir,
)
from aimet_onnx.experimental.llm_topology.topology_types import (
    BlockTopology,
    LinearGroup,
    LlmTopology,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)


def get_llm_topology(
    ir_model: onnx_ir.Model,
    block_boundaries: List[Tuple[str, str]],
    active_norms: Optional[List[ActiveNorm]] = None,
    active_norms_per_block: int = 2,
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
    topo_index: Optional[Dict[onnx_ir.Node, int]] = None,
) -> LlmTopology:
    """Build the LLM topology from pre-computed block boundaries.

    Per-block. Read groups are the weighted linears downstream of each active
    norm; write groups are found by walking back from the residual output
    tensor to the weighted linear(s) that feed it; the read groups are then
    split into fine-grained roles by module name, and the dynamic attention
    MatMuls are located by graph topology:

    * ``qkv``             — ``downstream_linears`` of the first active norm in
      the block (input norm), as a :class:`LinearGroup` split into
      ``q_proj`` / ``k_proj`` / ``v_proj`` (or ``FUSED_QKV``) by
      :func:`classify_linear_role`.
    * ``o_proj``          — weighted linear(s) that write the attention residual
      output (the post-attention norm input).
    * ``gate_up``         — ``downstream_linears`` of the second active norm in
      the block (post-attention norm), as a :class:`LinearGroup` split into
      ``gate_proj`` / ``up_proj`` (or ``FUSED_GATE_UP``).
    * ``down_proj``       — weighted linear(s) that write the block residual
      output (the block-end tensor).
    * ``qk_matmul`` / ``attn_v_matmul`` — the two dynamic (non-weighted)
      attention MatMuls between the ``qkv`` group and ``o_proj`` (see
      :func:`_find_attention_matmuls`).

    Model-level roles:

    * ``lm_head``         — downstream linears of active norms at or after the
      last block boundary (outside all decoder blocks).
    * ``embed_tokens``    — Gather ops with a static-weight input that appear
      before the first block boundary.

    :param ir_model: Analysis IR model from :func:`~.ir_analysis.build_analysis_ir`.
    :param block_boundaries: List of ``(start_tensor, end_tensor)`` residual-stream
        tensor names, as returned by :func:`get_decoder_block_boundaries`.
    :param active_norms: Active norms in topological order. Recomputed via
        :func:`find_active_norms_in_ir` when not supplied; pass a precomputed
        value to avoid a redundant graph scan.
    :param active_norms_per_block: Expected number of active norms per decoder
        block. Must match the value used in :func:`get_decoder_block_boundaries`.
        Defaults to 2 (Llama/Qwen2/Mistral/Phi family).
    :param role_patterns: Optional override of the default module-name → role
        table used to split the read groups (see :func:`classify_linear_role`).
    :param topo_index: Precomputed node → topological index map.
    :return: LlmTopology with block and backbone roles populated.
        ``hidden_size`` and ``head_dim`` are left ``None`` — use
        :func:`analyze_llm_topology` to also infer those.
    """
    if topo_index is None:
        topo_index = ir_analysis.topological_index(ir_model)
    if active_norms is None:
        active_norms = find_active_norms_in_ir(ir_model, topo_index)
    boundary_topo = ir_analysis.tensor_to_first_consumer_index(ir_model, topo_index)
    node_by_output = ir_analysis.node_by_output_tensor(ir_model)
    node_by_name = ir_analysis.node_by_name(ir_model)

    result = LlmTopology(active_norms=active_norms)

    for block_idx, (start_tensor, end_tensor) in enumerate(block_boundaries):
        start_topo = boundary_topo[start_tensor]
        end_topo = boundary_topo[end_tensor]

        # Active norms whose norm node falls in [start_topo, end_topo).
        # index 0 = input_norm (pre-attention),
        # index 1 = post_attn_norm (pre-MLP).
        block_active_norms = [
            active_norm
            for active_norm in active_norms
            if start_topo <= active_norm.topo_index < end_topo
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

        qkv = LinearGroup.classify(input_norm.downstream_linears, role_patterns)
        gate_up = LinearGroup.classify(post_attn_norm.downstream_linears, role_patterns)

        intermediate_tensor = post_attn_norm.input_tensor
        o_proj_candidates = _find_nearest_upstream_linears(
            intermediate_tensor, start_tensor, node_by_output, topo_index
        )
        if not o_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no attention residual writer (o_proj) found "
                f"for residual output '{intermediate_tensor}'."
            )

        down_proj_candidates = _find_nearest_upstream_linears(
            end_tensor, intermediate_tensor, node_by_output, topo_index
        )
        if not down_proj_candidates:
            raise ValueError(
                f"Block {block_idx}: no MLP residual writer (down_proj) found "
                f"for residual output '{end_tensor}'."
            )

        qk_matmul, attn_v_matmul = _find_attention_matmuls(
            [node_by_name[name] for name in qkv.linears],
            o_proj_candidates,
            topo_index,
        )

        block = BlockTopology(
            qkv=qkv,
            o_proj=ir_analysis.node_names(o_proj_candidates),
            gate_up=gate_up,
            down_proj=ir_analysis.node_names(down_proj_candidates),
            qk_matmul=ir_analysis.node_names(qk_matmul),
            attn_v_matmul=ir_analysis.node_names(attn_v_matmul),
            residual_input=start_tensor,
            residual_output=end_tensor,
        )
        result.blocks.append(block)
        _logger.debug(
            "Block %d: q=%s k=%s v=%s (fused_qkv=%s) o_proj=%s  gate=%s up=%s "
            "(fused_gate_up=%s) down_proj=%s  qk_matmul=%s attn_v_matmul=%s",
            block_idx,
            qkv.role(LinearRole.Q_PROJ),
            qkv.role(LinearRole.K_PROJ),
            qkv.role(LinearRole.V_PROJ),
            qkv.role(LinearRole.FUSED_QKV),
            block.o_proj,
            gate_up.role(LinearRole.GATE_PROJ),
            gate_up.role(LinearRole.UP_PROJ),
            gate_up.role(LinearRole.FUSED_GATE_UP),
            block.down_proj,
            block.qk_matmul,
            block.attn_v_matmul,
        )

    block_role_counts = [
        (
            len(b.qkv.linears),
            len(b.o_proj),
            len(b.gate_up.linears),
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
        linear
        for active_norm in active_norms
        if active_norm.topo_index >= last_end_topo
        for linear in active_norm.downstream_linears
    ]
    if not result.lm_head:
        _logger.debug(
            "lm_head not detected: no active norm found after the last block boundary."
        )
    else:
        _logger.debug("lm_head: %s", result.lm_head)

    first_start_topo = boundary_topo[block_boundaries[0][0]]
    result.embed_tokens = ir_analysis.node_names(
        [
            node
            for node in ir_model.graph
            if topo_index[node] < first_start_topo
            and node.op_type in ir_analysis.EMBEDDING_TYPES
            and _is_embedding_table_gather(node)
        ]
    )
    if not result.embed_tokens:
        _logger.info(
            "Backbone: embed_tokens not detected, no Gather op with a static weight found before "
            "the first block boundary. This is expected for VLM backbones exported with "
            "use_inputs_embeds=True. Rotate embedding.pth separately."
        )
    _logger.debug("embed_tokens: %s", result.embed_tokens)

    # Collected tolerantly: prefill-only / R1-only flows leave this empty and
    # never require KV-cache inputs. R3 validates the count against blocks.
    result.past_key_input_names = _collect_past_key_input_names_in_order(ir_model)
    _logger.debug("past_key inputs: %s", result.past_key_input_names)

    _logger.info(
        "Backbone: %d block(s), embed_tokens=%s, lm_head=%s.",
        len(result.blocks),
        result.embed_tokens,
        result.lm_head,
    )

    return result


def analyze_llm_topology(
    model: ModelProto,
    active_norms_per_block: int = 2,
    expected_num_blocks: Optional[int] = None,
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
    ir_model: Optional[onnx_ir.Model] = None,
) -> LlmTopology:
    """Analyze ``model`` end-to-end and return a name-based :class:`LlmTopology`.

    Runs the whole pipeline on a private onnx_ir copy of ``model``: strip
    quantizers, fuse RMSNorms, detect active norms and block boundaries, build
    the per-block topology, and infer ``hidden_size`` / ``head_dim``.

    :param model: ONNX ModelProto to analyze. Not mutated.
    :param active_norms_per_block: Active norms per decoder block (see
        :func:`get_decoder_block_boundaries`). Defaults to 2.
    :param expected_num_blocks: If given, validated against the detected count.
    :param role_patterns: Optional module-name → role override (see
        :func:`classify_linear_role`).
    :param ir_model: Pre-built *analysis* IR for ``model`` — as returned by
        :func:`~.ir_analysis.build_analysis_ir`, i.e. quantizer-stripped and
        RMSNorm-fused. Built here when ``None``. Pass one only to avoid a second
        ``from_proto`` of a large model when the caller already holds it; a
        faithful (unfused) IR will not detect norms and must not be passed.
    :return: LlmTopology with block/backbone roles, ``active_norms``,
        ``hidden_size`` and ``head_dim`` populated. ``head_dim`` is ``None`` when
        the export exposes no ``past_value`` graph input to derive it from.
    """
    if ir_model is None:
        ir_model = ir_analysis.build_analysis_ir(model)
    topo_index = ir_analysis.topological_index(ir_model)

    active_norms = find_active_norms_in_ir(ir_model, topo_index)
    boundaries = get_decoder_block_boundaries_in_ir(
        ir_model,
        active_norms=active_norms,
        expected_num_blocks=expected_num_blocks,
        active_norms_per_block=active_norms_per_block,
        topo_index=topo_index,
    )
    topology = get_llm_topology(
        ir_model,
        boundaries,
        active_norms=active_norms,
        active_norms_per_block=active_norms_per_block,
        role_patterns=role_patterns,
        topo_index=topo_index,
    )

    topology.hidden_size = _infer_hidden_size(ir_model, topology)

    # head_dim requires a KV-cache 'past_value' graph input; tolerate its
    # absence (prefill-only / R1-only flows do not need it).
    try:
        topology.head_dim = _infer_head_dim(model)
    except ValueError:
        topology.head_dim = None

    return topology


def _collect_past_key_input_names_in_order(ir_model: onnx_ir.Model) -> List[str]:
    """Return ``past_key_*`` graph input names in declaration order.

    HF/optimum LLM exports with a KV-cache expose one such input per decoder
    block. Prefill-only exports have none.
    """
    return [
        value.name
        for value in ir_model.graph.inputs
        if value.name and ("past_key" in value.name or "past_k_" in value.name)
    ]


def _is_embedding_table_gather(node: onnx_ir.Node) -> bool:
    """Return True if ``node`` is a token-embedding ``Gather`` (data is a 2-D table).

    A real embedding ``Gather`` has the embedding *table* as its first (data)
    input — a static rank-2 ``[vocab, hidden]`` initializer. Other Gathers in
    the prologue (e.g. position-id lookups, ``shape``-derived indexers) hold
    static scalar or 1-D constants on input 0 and must be excluded.

    :param node: Candidate Gather node.
    :return: True iff ``node`` looks like a token-embedding lookup.
    """
    if not node.inputs:
        return False
    table = static_tensor(node.inputs[0])
    if table is None:
        return False
    return len(table.shape) >= 2


def _find_attention_matmuls(
    qkv_linears: List[onnx_ir.Node],
    o_proj: List[onnx_ir.Node],
    topo_index: Dict[onnx_ir.Node, int],
) -> Tuple[List[onnx_ir.Node], List[onnx_ir.Node]]:
    """Return ``(qk_matmul_nodes, attn_v_matmul_nodes)`` for a decoder block.

    Attention computes ``softmax(Q @ Kᵀ / scale) @ V``. Both MatMuls are
    *dynamic* — both inputs are activations, so neither has a static weight.
    Walking forward from the QKV projections toward O (not crossing ``o_proj``
    or any other weighted linear), we collect every dynamic MatMul and every
    Softmax. A dynamic MatMul that *feeds* a Softmax is Q·Kᵀ; one that
    *consumes* a Softmax output is softmax·V.

    Per-head split (SHA) exports emit one of each per head, so both lists may
    hold multiple nodes. Returns empty lists when the pattern is absent (e.g. an
    export that fuses attention into a single op with no explicit MatMuls) —
    dynamic-MatMul identification is best-effort and not required by every
    consumer.

    :param qkv_linears: The block's Q/K/V projection nodes (walk start).
    :param o_proj: The block's attention-output projection node(s) (walk fence).
    :param topo_index: Node → topological index map, used to order the results.
    :return: Two lists of dynamic MatMul nodes: Q·Kᵀ and softmax·V.
    """
    o_proj_nodes = set(o_proj)
    dynamic_matmuls: List[onnx_ir.Node] = []
    visited = set()
    queue = [successor for linear in qkv_linears for successor in linear.successors()]
    while queue:
        node = queue.pop()
        if node in visited or node in o_proj_nodes:
            continue
        visited.add(node)
        # Do not cross other weighted linears — the attention path holds only
        # dynamic MatMuls between the QKV projections and O.
        if ir_analysis.is_weighted_linear(node):
            continue
        if ir_analysis.is_dynamic_matmul(node):
            dynamic_matmuls.append(node)
        queue.extend(node.successors())

    qk_matmul = [m for m in dynamic_matmuls if _matmul_touches_softmax(m, forward=True)]
    attn_v_matmul = [
        m for m in dynamic_matmuls if _matmul_touches_softmax(m, forward=False)
    ]
    return (
        ir_analysis.sorted_by_topology(qk_matmul, topo_index),
        ir_analysis.sorted_by_topology(attn_v_matmul, topo_index),
    )


def _matmul_touches_softmax(matmul: onnx_ir.Node, forward: bool) -> bool:
    """Return True if a Softmax is reachable from ``matmul`` in the given direction.

    Walks ``forward`` (through consumers) or backward (through input producers)
    from ``matmul``, stopping at the next MatMul boundary. A Softmax reached
    before hitting another MatMul means ``matmul`` feeds (forward) or consumes
    (backward) that Softmax — i.e. it is Q·Kᵀ or softmax·V respectively.
    """
    visited = set()
    queue = list(matmul.successors() if forward else matmul.predecessors())
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op_type in ir_analysis.SOFTMAX_TYPES:
            return True
        # Stop at any other MatMul so a head's Q·Kᵀ is not linked to the next
        # head's Softmax through a shared downstream op.
        if node.op_type == "MatMul":
            continue
        queue.extend(node.successors() if forward else node.predecessors())
    return False


def _find_nearest_upstream_linears(
    target_tensor: str,
    boundary_tensor: str,
    node_by_output: Dict[str, onnx_ir.Node],
    topo_index: Dict[onnx_ir.Node, int],
) -> List[onnx_ir.Node]:
    """Nearest weighted linears feeding target_tensor, via a backward walk.

    Walks backward from the node producing target_tensor and collects the first
    weighted linear (MatMul/Gemm/Conv with a static weight) on each path,
    stopping there; any other op type is crossed transparently.

    NOTE: The walk is fenced at boundary_tensor's producer: that node and anything
    earlier are skipped, so the walk does not cross into the previous block.

    :param target_tensor: Tensor whose upstream linears are wanted (walk start).
    :param boundary_tensor: Upstream edge; its producer and earlier nodes are the
        lower fence.
    :param node_by_output: Map of output tensor name -> producing node.
    :param topo_index: Map of node -> topological index.
    :return: The nearest weighted linear on each backward path, in topological order.
    """
    start = node_by_output.get(target_tensor)
    if start is None:
        return []

    fence_node = node_by_output.get(boundary_tensor)
    lo = topo_index[fence_node] if fence_node is not None else -1

    linears = []
    seen = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        if topo_index.get(node, -1) <= lo:
            continue
        if ir_analysis.is_weighted_linear(node):
            linears.append(node)
            continue  # this linear shadows everything upstream of it
        queue.extend(node.predecessors())
    return ir_analysis.sorted_by_topology(linears, topo_index)


def _infer_hidden_size(ir_model: onnx_ir.Model, role_map: LlmTopology) -> int:
    """Infer the model hidden size from embed_tokens, lm_head, or q/k/v_proj weights.

    Tries ``embed_tokens`` first (Gather table ``[vocab, hidden]``, last dim = hidden).
    Falls back to ``lm_head``, then to each block's ``qkv`` group, for backbones
    exported with ``use_inputs_embeds=True`` that have no Gather op.

    Takes the analysis IR rather than a ``ModelProto`` so the weight layout is
    derived by the one implementation that already knows it,
    :func:`~.ir_analysis.get_weight_value` — the topology itself only carries node
    names. Shapes are read off the static tensor without materializing it; an
    lm_head table can be hundreds of megabytes.

    :param ir_model: Analysis IR model from :func:`~.ir_analysis.build_analysis_ir`.
    :param role_map: Topology produced by :func:`get_llm_topology`.
    :return: The hidden dimension size.
    """
    node_by_name = ir_analysis.node_by_name(ir_model)

    for embed_name in role_map.embed_tokens:
        # Only the data input (a [vocab, hidden] table) yields hidden_size; other
        # static inputs (e.g. axis attributes, indices) are not embedding tables.
        node = node_by_name.get(embed_name)
        table = static_tensor(node.inputs[0]) if node else None
        if table is not None and len(table.shape) >= 2:
            return int(table.shape[-1])

    # Gemm transB=1 stores W [vocab, hidden] -> hidden = shape[-1].
    # MatMul stores W [hidden, vocab]        -> hidden = shape[0].
    # Conv 1x1 stores W [vocab, hidden, 1, 1] -> hidden = shape[1].
    for linear_name in [
        *role_map.lm_head,
        *(name for block in role_map.blocks for name in block.qkv.linears),
    ]:
        node = node_by_name.get(linear_name)
        if node is None:
            continue
        weight, is_transposed = ir_analysis.get_weight_value(node)
        if weight is None:
            continue
        shape = static_tensor(weight).shape
        if node.op_type == "Conv":
            return int(shape[1])  # [out_ch, in_ch, *k]: in_ch = hidden
        return int(shape[-1] if is_transposed else shape[0])

    raise ValueError(
        "Cannot infer hidden_size: no embed_tokens, lm_head or qkv_proj static weight found in role_map"
    )


def _infer_head_dim(model: ModelProto) -> int:
    """Infer per-head dimension from a ``past_value`` graph input's last axis.

    HF/optimum LLM exports include ``past_value_*`` (or ``past_key_values.*.value``)
    inputs whose final dimension is ``head_dim`` regardless of the surrounding
    layout (``[B, num_kv_heads, past_seq, head_dim]`` or
    ``[B, past_seq, num_kv_heads, head_dim]``). This avoids having to derive
    ``head_dim`` from ``hidden_size / num_heads``, which is wrong for models
    that decouple the two (e.g. Gemma3 fixes ``head_dim=256`` independent of
    hidden size).

    :param model: ONNX ModelProto whose graph inputs are scanned.
    :return: ``head_dim`` read from the last dim of the first matching input.
    :raises ValueError: If no ``past_value`` input exists, or if its last dim
        is not a static positive integer.
    """
    for inp in model.graph.input:
        if "past_value" not in inp.name:
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 0:
            continue
        last = dims[-1]
        # Must be a statically-known positive int. Symbolic dims (dim_param set,
        # or dim_value == 0) cannot be used to derive head_dim.
        if last.HasField("dim_value") and last.dim_value > 0:
            head_dim = last.dim_value
            _logger.info(
                "Derived head_dim=%d from graph input '%s' (last dim of shape %s).",
                head_dim,
                inp.name,
                [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims],
            )
            return head_dim

    raise ValueError(
        "Cannot infer head_dim: no graph input matching 'past_value' with a "
        "static positive last dimension was found."
    )


__all__ = [
    "analyze_llm_topology",
    "get_llm_topology",
]
