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
specific quantization technique. Everything here works in ConnectedGraph
``Op`` space; techniques that must mutate raw ``NodeProto`` edges (e.g.
SpinQuant R3) derive their insertion anchors from this topology separately.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph, Product
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
    tensor_to_first_consumer_index,
)
from aimet_onnx.experimental.llm_topology.layer_roles import (
    LinearRole,
    classify_linear_role,
)
from aimet_onnx.experimental.llm_topology.norm_detection import (
    ActiveNorm,
    find_active_norms,
)
from aimet_onnx.experimental.llm_topology.weight_utils import (
    get_weight_product,
    infer_head_dim,
    infer_hidden_size,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)

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
class LinearGroup:
    """A norm's downstream weighted linears, together with their role split.

    ``ops`` is the coarse read group (the single source of truth): every
    weighted linear reading through one active norm. ``by_role`` is a
    name-based *partition* of ``ops`` produced by :func:`classify_linear_role`
    — each op appears under exactly one :class:`LinearRole` (unmatched ops fall
    under :attr:`LinearRole.UNKNOWN`). Because the split is derived from ``ops``
    at construction, the two can never disagree.

    Consumers that treat the whole group uniformly (e.g. an R1 residual-stream
    rotation) iterate ``ops``; consumers that touch one projection (e.g. R2
    rotates only V) read :meth:`role`. A role maps to a *list* because per-head
    split exports (SHA) emit one op per head, and fused exports (e.g. Phi3
    ``qkv_proj``) collapse several logical roles into a single op reported under
    a ``FUSED_*`` role.

    :param ops: All weighted linears reading through one active norm.
    :param by_role: Partition of ``ops`` keyed by :class:`LinearRole`.
    """

    ops: List[Op] = field(default_factory=list)
    by_role: Dict[LinearRole, List[Op]] = field(default_factory=dict)

    @classmethod
    def classify(
        cls,
        ops: List[Op],
        role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
    ) -> "LinearGroup":
        """Build a group from ``ops``, splitting it into roles by module name."""
        return cls(ops=list(ops), by_role=_split_by_role(ops, role_patterns))

    def role(self, role: LinearRole) -> List[Op]:
        """Return the ops classified as ``role`` (empty list if none)."""
        return self.by_role.get(role, [])


@dataclass
class BlockTopology:
    """Topology of a single decoder block: weighted projections + dynamic MatMuls.

    The two weighted read groups are :class:`LinearGroup` values — each exposes
    both its coarse ``ops`` list and the fine-grained role split (see
    :class:`LinearGroup` and the ``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``gate_proj`` / ``up_proj`` convenience properties below). The two write
    projections and the dynamic attention MatMuls are plain op lists.

    :param qkv: Attention read group — the Q/K/V (or fused QKV) projections
        reading through the block's input norm.
    :param o_proj: Attention-output projection op(s) writing to the residual.
    :param gate_up: MLP read group — the gate/up (or fused gate-up) projections
        reading through the post-attention norm.
    :param down_proj: MLP-output projection op(s) writing to the residual.
    :param qk_matmul: The dynamic (non-weighted) Q·Kᵀ attention MatMul op(s) —
        one per query head in SHA exports.
    :param attn_v_matmul: The dynamic (non-weighted) softmax·V MatMul op(s).
    :param residual_input: Residual-stream tensor entering the block's input norm.
    :param residual_output: Residual-stream tensor leaving the block.
    """

    qkv: LinearGroup = field(default_factory=LinearGroup)
    o_proj: List[Op] = field(default_factory=list)
    gate_up: LinearGroup = field(default_factory=LinearGroup)
    down_proj: List[Op] = field(default_factory=list)

    qk_matmul: List[Op] = field(default_factory=list)
    attn_v_matmul: List[Op] = field(default_factory=list)

    residual_input: Optional[Product] = None
    residual_output: Optional[Product] = None

    @property
    def q_proj(self) -> List[Op]:
        """Query projection op(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.Q_PROJ)

    @property
    def k_proj(self) -> List[Op]:
        """Key projection op(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.K_PROJ)

    @property
    def v_proj(self) -> List[Op]:
        """Value projection op(s), split from ``qkv`` by module name."""
        return self.qkv.role(LinearRole.V_PROJ)

    @property
    def gate_proj(self) -> List[Op]:
        """Gate projection op(s), split from ``gate_up`` by module name."""
        return self.gate_up.role(LinearRole.GATE_PROJ)

    @property
    def up_proj(self) -> List[Op]:
        """Up projection op(s), split from ``gate_up`` by module name."""
        return self.gate_up.role(LinearRole.UP_PROJ)


@dataclass
class LlmTopology:
    """Topology of an ONNX decoder-stack model: blocks + backbone-level roles + dims.

    :param embed_tokens: Token-embedding Gather op(s) that produce the initial
        residual-stream activations.
    :param lm_head: Vocabulary-projection linear(s) downstream of the final norm.
    :param blocks: Per-decoder-block topology in topological order.
    :param past_key_input_names: Raw ``past_key_*`` graph inputs in declaration
        order, collected tolerantly (empty for prefill-only exports without a
        KV-cache). Pairing these to ``blocks`` and validating that their count
        matches the block count are the consumer's responsibility (e.g. R3) —
        R1-only and prefill-only flows do not require KV-cache inputs.
    :param active_norms: Active norms in topological order used to build the
        topology (``None`` when the topology was built from a pre-supplied list
        that the builder did not retain).
    :param hidden_size: Residual-stream hidden dimension (``None`` if not
        inferred; :func:`analyze_llm_topology` fills it).
    :param head_dim: Per-head dimension (``None`` when it could not be derived,
        e.g. an export without KV-cache inputs).
    """

    embed_tokens: List[Op] = field(default_factory=list)
    lm_head: List[Op] = field(default_factory=list)
    blocks: List[BlockTopology] = field(default_factory=list)
    past_key_input_names: List[str] = field(default_factory=list)
    active_norms: Optional[List[ActiveNorm]] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None


def get_llm_topology(
    connected_graph: ConnectedGraph,
    block_boundaries: List[Tuple[str, str]],
    active_norms: Optional[List[ActiveNorm]] = None,
    active_norms_per_block: int = 2,
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
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

    :param connected_graph: ConnectedGraph built from the model.
    :param block_boundaries: List of ``(start_tensor, end_tensor)`` residual-stream
        tensor names, as returned by :func:`get_decoder_block_boundaries`.
    :param active_norms: Active norms in topological order. Recomputed via
        :func:`find_active_norms` when not supplied; pass a precomputed value to
        avoid a redundant graph scan.
    :param active_norms_per_block: Expected number of active norms per decoder
        block. Must match the value used in :func:`get_decoder_block_boundaries`.
        Defaults to 2 (Llama/Qwen2/Mistral/Phi family).
    :param role_patterns: Optional override of the default module-name → role
        table used to split the read groups (see :func:`classify_linear_role`).
    :return: LlmTopology with block and backbone roles populated. ``hidden_size``
        and ``head_dim`` are left ``None`` — use :func:`analyze_llm_topology`
        to also infer those.
    """
    op_topo_idx = {op: i for i, op in enumerate(connected_graph.ordered_ops)}
    if active_norms is None:
        active_norms = find_active_norms(connected_graph.model, connected_graph)
    boundary_topo = tensor_to_first_consumer_index(connected_graph)

    # tensor name -> producing Op
    producer_by_tensor = {
        out.name: op for op in connected_graph.ordered_ops for out in op.outputs
    }

    result = LlmTopology(active_norms=active_norms)

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

        qkv = LinearGroup.classify(input_norm.downstream_linears, role_patterns)
        gate_up = LinearGroup.classify(post_attn_norm.downstream_linears, role_patterns)

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

        qk_matmul, attn_v_matmul = _find_attention_matmuls(qkv.ops, o_proj_candidates)

        result.blocks.append(
            BlockTopology(
                qkv=qkv,
                o_proj=o_proj_candidates,
                gate_up=gate_up,
                down_proj=down_proj_candidates,
                qk_matmul=qk_matmul,
                attn_v_matmul=attn_v_matmul,
                residual_input=connected_graph.get_product(start_tensor),
                residual_output=connected_graph.get_product(end_tensor),
            )
        )
        _logger.debug(
            "Block %d: q=%s k=%s v=%s (fused_qkv=%s) o_proj=%s  gate=%s up=%s "
            "(fused_gate_up=%s) down_proj=%s  qk_matmul=%s attn_v_matmul=%s",
            block_idx,
            [op.name for op in qkv.role(LinearRole.Q_PROJ)],
            [op.name for op in qkv.role(LinearRole.K_PROJ)],
            [op.name for op in qkv.role(LinearRole.V_PROJ)],
            [op.name for op in qkv.role(LinearRole.FUSED_QKV)],
            [op.name for op in o_proj_candidates],
            [op.name for op in gate_up.role(LinearRole.GATE_PROJ)],
            [op.name for op in gate_up.role(LinearRole.UP_PROJ)],
            [op.name for op in gate_up.role(LinearRole.FUSED_GATE_UP)],
            [op.name for op in down_proj_candidates],
            [op.name for op in qk_matmul],
            [op.name for op in attn_v_matmul],
        )

    block_role_counts = [
        (
            len(b.qkv.ops),
            len(b.o_proj),
            len(b.gate_up.ops),
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
        for op in connected_graph.ordered_ops[:first_start_topo]
        if op.type in _EMBEDDING_TYPES and _is_embedding_table_gather(op)
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


def analyze_llm_topology(
    model: ModelProto,
    connected_graph: Optional[ConnectedGraph] = None,
    active_norms_per_block: int = 2,
    expected_num_blocks: Optional[int] = None,
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
) -> LlmTopology:
    """Analyze ``model`` end-to-end and return a fully-populated :class:`LlmTopology`.

    Convenience facade that runs the whole pipeline: build the ConnectedGraph
    (if not supplied), detect active norms and block boundaries, build the
    per-block topology, and infer ``hidden_size`` / ``head_dim``. Callers that
    already hold a ConnectedGraph and/or intermediate results can call
    :func:`get_decoder_block_boundaries` + :func:`get_llm_topology` directly.

    :param model: ONNX ModelProto to analyze.
    :param connected_graph: Pre-built ConnectedGraph for ``model``; built here
        when ``None``.
    :param active_norms_per_block: Active norms per decoder block (see
        :func:`get_decoder_block_boundaries`). Defaults to 2.
    :param expected_num_blocks: If given, validated against the detected count.
    :param role_patterns: Optional module-name → role override (see
        :func:`classify_linear_role`).
    :return: LlmTopology with block/backbone roles, ``active_norms``,
        ``hidden_size`` and ``head_dim`` populated. ``head_dim`` is ``None`` when
        the export exposes no ``past_value`` graph input to derive it from.
    """
    if connected_graph is None:
        connected_graph = ConnectedGraph(model)

    active_norms = find_active_norms(model, connected_graph)
    boundaries = get_decoder_block_boundaries(
        model,
        connected_graph,
        expected_num_blocks=expected_num_blocks,
        active_norms_per_block=active_norms_per_block,
    )
    topology = get_llm_topology(
        connected_graph,
        boundaries,
        active_norms=active_norms,
        active_norms_per_block=active_norms_per_block,
        role_patterns=role_patterns,
    )

    topology.hidden_size = infer_hidden_size(model, topology)

    # head_dim requires a KV-cache 'past_value' graph input; tolerate its
    # absence (prefill-only / R1-only flows do not need it).
    try:
        topology.head_dim = infer_head_dim(model)
    except ValueError:
        topology.head_dim = None

    return topology


def _split_by_role(
    linears: List[Op],
    role_patterns: Optional[Dict[LinearRole, Pattern]],
) -> Dict[LinearRole, List[Op]]:
    """Classify each op in ``linears`` by module name into its fine-grained role.

    :param linears: A read group (``qkv_linears`` or ``gate_up_linears``).
    :param role_patterns: Optional override passed to :func:`classify_linear_role`.
    :return: A mapping from every :class:`LinearRole` to the (possibly empty)
        list of ops classified as that role. Ops that do not match any role are
        logged and dropped from the split (they remain in the coarse group).
    """
    out: Dict[LinearRole, List[Op]] = {role: [] for role in LinearRole}
    for op in linears:
        role = classify_linear_role(op, role_patterns)
        out[role].append(op)
        if role is LinearRole.UNKNOWN:
            _logger.debug(
                "Linear '%s' did not match any known role pattern; left "
                "un-split (still present in its coarse read group).",
                op.name,
            )
    return out


_SOFTMAX_TYPES = frozenset(("Softmax",))


def _is_dynamic_matmul(op: Op) -> bool:
    """Return True if ``op`` is a MatMul with no static weight (both inputs dynamic)."""
    return op.type == "MatMul" and get_weight_product(op)[0] is None


def _find_attention_matmuls(
    qkv_linears: List[Op],
    o_proj: List[Op],
) -> Tuple[List[Op], List[Op]]:
    """Return ``(qk_matmul_ops, attn_v_matmul_ops)`` for a decoder block.

    Attention computes ``softmax(Q @ Kᵀ / scale) @ V``. Both MatMuls are
    *dynamic* — both inputs are activations, so neither has a static weight.
    Walking forward from the QKV projections toward O (not crossing ``o_proj``
    or any other weighted linear), we collect every dynamic MatMul and every
    Softmax. A dynamic MatMul that *feeds* a Softmax is Q·Kᵀ; one that
    *consumes* a Softmax output is softmax·V.

    Per-head split (SHA) exports emit one of each per head, so both lists may
    hold multiple ops. Returns empty lists when the pattern is absent (e.g. an
    export that fuses attention into a single op with no explicit MatMuls) —
    dynamic-MatMul identification is best-effort and not required by every
    consumer.

    :param qkv_linears: The block's Q/K/V projection ops (walk start).
    :param o_proj: The block's attention-output projection op(s) (walk fence).
    :return: Two lists of dynamic MatMul ops: Q·Kᵀ and softmax·V.
    """
    o_proj_set = set(o_proj)
    dynamic_matmuls: List[Op] = []
    visited: set = set()
    queue = [consumer for lin in qkv_linears for consumer in lin.output_ops]
    while queue:
        op = queue.pop()
        if op in visited or op in o_proj_set:
            continue
        visited.add(op)
        # Do not cross other weighted linears — the attention path holds only
        # dynamic MatMuls between the QKV projections and O.
        if op.type in _LINEAR_TYPES and get_weight_product(op)[0] is not None:
            continue
        if _is_dynamic_matmul(op):
            dynamic_matmuls.append(op)
        queue.extend(op.output_ops)

    qk_matmul = [m for m in dynamic_matmuls if _matmul_touches_softmax(m, forward=True)]
    attn_v_matmul = [
        m for m in dynamic_matmuls if _matmul_touches_softmax(m, forward=False)
    ]
    return qk_matmul, attn_v_matmul


def _matmul_touches_softmax(matmul: Op, forward: bool) -> bool:
    """Return True if a Softmax is reachable from ``matmul`` in the given direction.

    Walks ``forward`` (through ``output_ops``) or backward (through
    ``input_ops``) from ``matmul``, stopping at the next MatMul boundary. A
    Softmax reached before hitting another MatMul means ``matmul`` feeds
    (forward) or consumes (backward) that Softmax — i.e. it is Q·Kᵀ or
    softmax·V respectively.
    """
    visited: set = set()
    queue = list(matmul.output_ops if forward else matmul.input_ops)
    while queue:
        op = queue.pop()
        if op in visited:
            continue
        visited.add(op)
        if op.type in _SOFTMAX_TYPES:
            return True
        # Stop at any other MatMul so a head's Q·Kᵀ is not linked to the next
        # head's Softmax through a shared downstream op.
        if op.type == "MatMul":
            continue
        queue.extend(op.output_ops if forward else op.input_ops)
    return False


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
