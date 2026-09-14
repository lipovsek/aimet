# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ConnectedGraph-flavored view of an LLM topology. TRANSITIONAL.

The analyzers in this package describe a decoder stack by name (see
:mod:`~.topology_by_name`). SpinQuant's rotation passes still expect
ConnectedGraph ``Op`` objects, so :func:`resolve_topology` re-attaches a
:class:`~aimet_onnx.meta.connectedgraph.ConnectedGraph` to a name-based topology
and hands back the ``Op``-bearing dataclasses those passes were written against.

This whole module exists only to keep those consumers working while they migrate
onto the name-based topology; it is expected to be deleted, not extended. New
code should consume :class:`~.topology_by_name.LlmTopologyByName` directly.

Resolution keys on node names, which ``ConnectedGraph._ops`` is already indexed
by. :func:`~.ir_analysis.node_name` rejects unnamed nodes during analysis, so a
name is always present by the time anything reaches here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aimet_onnx.meta.connectedgraph import ConnectedGraph, Product
from aimet_onnx.meta.operations import Op

from aimet_onnx.experimental.llm_topology.layer_roles import LinearRole
from aimet_onnx.experimental.llm_topology.norm_detection import ActiveNormByName
from aimet_onnx.experimental.llm_topology.topology_by_name import (
    LinearGroupByName,
    LlmTopologyByName,
)


@dataclass
class ActiveNorm:
    """An affine RMSNorm that has at least one downstream weight linear op.

    The ``Op``-bearing counterpart of
    :class:`~.norm_detection.ActiveNormByName`. SpinQuant's R1 absorbs the norm's
    gamma into these linears and reads their weight layout straight off the
    ``Op``, so the weight-layout logic stays in one place
    (:func:`~.weight_utils.get_weight_product`) rather than being re-derived here.

    :param scale_name: Name of the gamma (scale) initializer in the model.
    :param downstream_linears: MatMul/Gemm/Conv ops reachable from the norm.
    :param input_tensor: Residual-stream tensor entering the norm.
    """

    scale_name: str
    downstream_linears: List[Op] = field(default_factory=list)
    input_tensor: str = ""


@dataclass
class LinearGroup:
    """A norm's downstream weighted linears, together with their role split.

    ``ops`` is the coarse read group (the single source of truth): every
    weighted linear reading through one active norm. ``by_role`` is a
    name-based *partition* of ``ops`` — each op appears under exactly one
    :class:`LinearRole` (unmatched ops fall under :attr:`LinearRole.UNKNOWN`).

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
        topology. These are name-based
        (:class:`~.norm_detection.ActiveNorm`) even here — nothing consumes them
        as ConnectedGraph ops.
    :param hidden_size: Residual-stream hidden dimension (``None`` if not
        inferred; :func:`~.topology.analyze_llm_topology` fills it).
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


def resolve_topology(
    topology: LlmTopologyByName,
    connected_graph: ConnectedGraph,
) -> LlmTopology:
    """Re-attach ``connected_graph`` to a name-based topology.

    Every node name in the topology is replaced with the ConnectedGraph ``Op`` of
    that name; residual-stream tensor names become ``Product`` objects.

    :param topology: Name-based topology, as returned by
        :func:`~.topology.get_llm_topology`.
    :param connected_graph: ConnectedGraph built from the same model the topology
        was analyzed from.
    :return: The equivalent ``Op``-bearing :class:`LlmTopology`.
    :raises ValueError: If any name cannot be resolved against
        ``connected_graph`` — the two were built from different graphs.
    """
    op_by_name = {op.name: op for op in connected_graph.ordered_ops}

    resolved = LlmTopology(
        embed_tokens=_resolve_all(topology.embed_tokens, op_by_name),
        lm_head=_resolve_all(topology.lm_head, op_by_name),
        past_key_input_names=list(topology.past_key_input_names),
        active_norms=resolve_active_norms(topology.active_norms or [], connected_graph),
        hidden_size=topology.hidden_size,
        head_dim=topology.head_dim,
    )
    for block in topology.blocks:
        resolved.blocks.append(
            BlockTopology(
                qkv=_resolve_group(block.qkv, op_by_name),
                o_proj=_resolve_all(block.o_proj, op_by_name),
                gate_up=_resolve_group(block.gate_up, op_by_name),
                down_proj=_resolve_all(block.down_proj, op_by_name),
                qk_matmul=_resolve_all(block.qk_matmul, op_by_name),
                attn_v_matmul=_resolve_all(block.attn_v_matmul, op_by_name),
                residual_input=_resolve_product(block.residual_input, connected_graph),
                residual_output=_resolve_product(
                    block.residual_output, connected_graph
                ),
            )
        )
    return resolved


def resolve_active_norms(
    active_norms: List[ActiveNormByName],
    connected_graph: ConnectedGraph,
) -> List[ActiveNorm]:
    """Re-attach ``connected_graph`` to name-based active norms.

    :param active_norms: Name-based norms from
        :func:`~.norm_detection.find_active_norms`.
    :param connected_graph: ConnectedGraph built from the same model.
    :return: The equivalent ``Op``-bearing :class:`ActiveNorm`\\ s, in the same order.
    :raises ValueError: If a downstream linear cannot be resolved.
    """
    op_by_name = {op.name: op for op in connected_graph.ordered_ops}
    return [
        ActiveNorm(
            scale_name=active_norm.scale_name,
            downstream_linears=_resolve_all(active_norm.downstream_linears, op_by_name),
            input_tensor=active_norm.input_tensor,
        )
        for active_norm in active_norms
    ]


def _resolve_group(
    group: LinearGroupByName,
    op_by_name: Dict[str, Op],
) -> LinearGroup:
    """Resolve a name-based read group, preserving its role split."""
    return LinearGroup(
        ops=_resolve_all(group.linears, op_by_name),
        by_role={
            role: _resolve_all(names, op_by_name)
            for role, names in group.by_role.items()
        },
    )


def _resolve_all(node_names: List[str], op_by_name: Dict[str, Op]) -> List[Op]:
    """Resolve every node name to its ConnectedGraph ``Op``."""
    resolved = []
    for name in node_names:
        op = op_by_name.get(name)
        if op is None:
            raise ValueError(
                f"Cannot resolve node '{name}' against the supplied ConnectedGraph. "
                "The topology and the ConnectedGraph were built from different graphs."
            )
        resolved.append(op)
    return resolved


def _resolve_product(
    tensor_name: Optional[str],
    connected_graph: ConnectedGraph,
) -> Optional[Product]:
    """Resolve a residual-stream tensor name to its ConnectedGraph ``Product``.

    :raises ValueError: If the tensor is absent from ``connected_graph``. Returning
        ``None`` here would push the failure downstream into a rotation pass, which
        can only report it as a missing residual edge.
    """
    if tensor_name is None:
        return None
    product = connected_graph.get_product(tensor_name)
    if product is None:
        raise ValueError(
            f"Residual tensor '{tensor_name}' has no ConnectedGraph Product. "
            "The topology and the ConnectedGraph were built from different graphs."
        )
    return product


__all__ = [
    "ActiveNorm",
    "BlockTopology",
    "LinearGroup",
    "LlmTopology",
    "resolve_active_norms",
    "resolve_topology",
]
