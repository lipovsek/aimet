# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Anchor detection for SpinQuant R3 rotation, pinned by ``past_key_*`` inputs.

Per decoder block, R3 needs two insertion anchors:

* the K-side: the current-K input of the ``Concat`` that fuses ``past_key_*``
  with the current-K tensor (so K entering the cache is rotated);
* the Q-side: the Q input of the QK^T MatMul (so the rotations algebraically
  cancel inside ``Q @ K^T``).

We pin the search on the ``past_key_*`` graph inputs (HF/optimum LLM exports
expose one per decoder block, in declaration order). For each one:

1. Find the unique ``Concat`` consuming it. The OTHER input of that Concat
   is the current-K tensor (R3 K-side anchor).
2. Walk forward from the Concat output through pass-through ops
   (quantsim's grid-preserving data-movement ops plus ``Cast``; see
   :func:`_is_passthrough`) until reaching a unique ``MatMul`` — that's QK^T.
   Whichever MatMul input traces back to the Concat is K; the OTHER input is Q
   (R3 Q-side anchor).

R3 in this iteration requires KV-cache-style exports: the model must expose
one ``past_key_*`` graph input per decoder block. Prefill-only exports
without KV-cache inputs are not supported.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import onnx

from aimet_onnx.common.onnx._utils import _is_grid_preserving_op
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.utils import ModelProto

from aimet_onnx.experimental.spinquant.model_analysis.block_identifier import (
    DecoderModelRoleMap,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def _is_passthrough(op_type: str, domain: str = "") -> bool:
    """Return True for ops the R3 anchor search treats as transparent.

    These are quantsim's grid-preserving (data-movement) ops — ``Transpose``,
    ``Reshape``, ``Identity``, ``Unsqueeze``, ``Expand``, etc. — plus ``Cast``.
    ``Unsqueeze`` / ``Expand`` are the ops GQA's ``repeat_kv`` inserts to
    broadcast each KV head across its query-head group; they form a clean
    linear chain (``Unsqueeze -> Expand -> Reshape``) between the past-key
    Concat and QK^T, so walking through them reaches the same QK^T MatMul as
    the MHA case. R3 rotates the per-head ``head_dim`` axis, which is identical
    for the KV heads before and after this broadcast, so the rotation is
    unaffected.

    ``Cast`` is not grid-preserving in a quantized graph (it changes dtype), so
    it is absent from :func:`_is_grid_preserving_op`. But R3 runs on the float
    graph before quantization, and HF/optimum exports insert dtype Casts on the
    Q/K paths that must be walked through — hence the explicit addition here.
    """
    return op_type == "Cast" or _is_grid_preserving_op(op_type, domain)


@dataclass
class BlockR3Anchors:
    """Per-block anchors for inserting R3 online Hadamards.

    Members are raw ONNX ``NodeProto`` objects so the rotation pass can
    rewire ``node.input[idx]`` directly without going through the
    ConnectedGraph (which becomes stale once we insert new nodes anyway).

    :param past_key_input_name: The ``past_key_*`` graph input pinned this
        block's anchor search.
    :param k_concat_node: ``Concat`` node fusing past_key with current-K.
        R3 rewires its ``input[k_consumer_input_idx]``.
    :param k_consumer_input_idx: Index in ``k_concat_node.input`` currently
        holding the post-RoPE current-K tensor.
    :param k_input_tensor: Tensor name at
        ``k_concat_node.input[k_consumer_input_idx]``.
    :param qk_matmul_node: The QK^T attention MatMul reached forward from
        ``k_concat_node``. R3 rewires its ``input[q_consumer_input_idx]``.
    :param q_consumer_input_idx: Index in ``qk_matmul_node.input`` currently
        holding the post-RoPE Q tensor (the input that does NOT trace back
        to the Concat).
    :param q_input_tensor: Tensor name at
        ``qk_matmul_node.input[q_consumer_input_idx]``.
    """

    past_key_input_name: str
    k_concat_node: onnx.NodeProto
    k_consumer_input_idx: int
    k_input_tensor: str
    qk_matmul_node: onnx.NodeProto
    q_consumer_input_idx: int
    q_input_tensor: str


def find_r3_anchors(
    role_map: DecoderModelRoleMap, model: ModelProto
) -> List[BlockR3Anchors]:
    """Return per-block R3 anchors, pinned by ``past_key_*`` graph inputs.

    The number of past_key inputs in the model must equal the number of
    decoder blocks in ``role_map``. The two are paired by graph-input
    declaration order, which matches HF/optimum export conventions.

    :raises ValueError: If past_key input count does not match block count;
        if any past_key input is not consumed by exactly one Concat; if the
        forward walk from a Concat to its QK^T MatMul is ambiguous; or if
        the same MatMul ends up matched by two different blocks.
    """
    past_key_input_names = role_map.past_key_input_names
    n_blocks = len(role_map.blocks)
    if len(past_key_input_names) != n_blocks:
        raise ValueError(
            f"R3 rotation: expected {n_blocks} past_key_* graph inputs "
            f"(one per decoder block), found {len(past_key_input_names)}: "
            f"{past_key_input_names}. R3 requires a KV-cache-style export."
        )

    consumers_by_tensor = _index_consumers_by_tensor_name(model)
    producer_by_tensor = _index_producer_by_tensor_name(model)

    seen_matmul_ids: Set[int] = set()
    result: List[BlockR3Anchors] = []
    for block_idx, past_key_name in enumerate(past_key_input_names):
        _logger.debug(
            "R3 anchors: processing block %d (past_key='%s').",
            block_idx,
            past_key_name,
        )
        concat_node = _find_unique_concat_consumer(consumers_by_tensor, past_key_name)
        k_consumer_input_idx, k_input_tensor = _find_current_k_input_of_concat(
            concat_node, past_key_name, producer_by_tensor
        )

        qk_matmul_node = _walk_forward_to_unique_matmul(
            concat_node, consumers_by_tensor
        )
        if id(qk_matmul_node) in seen_matmul_ids:
            raise ValueError(
                f"R3 rotation: past_key input '{past_key_name}': QK^T MatMul "
                f"'{qk_matmul_node.name}' was already matched by an earlier "
                f"block. past_key_* graph inputs may be misordered."
            )
        seen_matmul_ids.add(id(qk_matmul_node))

        q_consumer_input_idx, q_input_tensor = _find_q_input_of_qk_matmul(
            qk_matmul_node, concat_node, producer_by_tensor
        )

        result.append(
            BlockR3Anchors(
                past_key_input_name=past_key_name,
                k_concat_node=concat_node,
                k_consumer_input_idx=k_consumer_input_idx,
                k_input_tensor=k_input_tensor,
                qk_matmul_node=qk_matmul_node,
                q_consumer_input_idx=q_consumer_input_idx,
                q_input_tensor=q_input_tensor,
            )
        )
    _logger.debug("R3 anchors resolved for %d block(s).", len(result))
    return result


def _index_consumers_by_tensor_name(
    model: ModelProto,
) -> Dict[str, List[onnx.NodeProto]]:
    out: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                out.setdefault(inp, []).append(node)
    return out


def _index_producer_by_tensor_name(
    model: ModelProto,
) -> Dict[str, onnx.NodeProto]:
    out: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for o in node.output:
            if o:
                out[o] = node
    return out


def _find_unique_concat_consumer(
    consumers_by_tensor: Dict[str, List[onnx.NodeProto]],
    past_key_name: str,
) -> onnx.NodeProto:
    """Return the unique Concat downstream of ``past_key_name``.

    Walk through pass-through ops (see :func:`_is_passthrough`) that may sit
    between the graph input and the underlying Concat.
    """
    visited: Set[str] = set()
    queue: deque = deque([past_key_name])
    while queue:
        name = queue.popleft()
        if name in visited:
            continue
        visited.add(name)
        consumers = consumers_by_tensor.get(name, [])
        for n in consumers:
            if n.op_type == "Concat":
                return n
            if _is_passthrough(n.op_type, n.domain):
                queue.extend(n.output)
    consumers = consumers_by_tensor.get(past_key_name, [])
    consumer_summary = [(n.name, n.op_type) for n in consumers]
    raise ValueError(
        f"R3 rotation: no downstream Concat reachable from past_key input "
        f"'{past_key_name}' through pass-through ops "
        f"(direct consumers: {consumer_summary})."
    )


def _find_current_k_input_of_concat(
    concat_node: onnx.NodeProto,
    past_key_name: str,
    producer_by_tensor: Dict[str, onnx.NodeProto],
) -> Tuple[int, str]:
    """Return ``(input_idx, tensor_name)`` of the current-K input of ``concat_node``.

    The past-key input may not be ``past_key_name`` directly: data-movement
    ops (Cast / Identity) can sit between the graph input and the Concat. We
    identify the past-key-side input by tracing backward through pass-through
    ops to the graph input ``past_key_name``.
    """
    past_indices = []
    cur_indices = []
    past_key_set = {past_key_name}
    for i, name in enumerate(concat_node.input):
        if _input_traces_back_to(name, past_key_set, producer_by_tensor):
            past_indices.append(i)
        else:
            cur_indices.append(i)
    if len(cur_indices) != 1:
        raise ValueError(
            f"R3 rotation: past_key input '{past_key_name}': Concat "
            f"'{concat_node.name}' has {len(cur_indices)} non-past_key inputs; "
            f"expected 1 (inputs={list(concat_node.input)}, "
            f"past_indices={past_indices})."
        )
    idx = cur_indices[0]
    return idx, concat_node.input[idx]


def _walk_forward_to_unique_matmul(
    start_node: onnx.NodeProto,
    consumers_by_tensor: Dict[str, List[onnx.NodeProto]],
) -> onnx.NodeProto:
    """Walk forward from ``start_node`` through pass-through ops until a unique MatMul.

    Steps only through ops accepted by :func:`_is_passthrough`. If at any
    step the consumers fan out ambiguously (multiple pass-through consumers,
    or zero consumers, or multiple MatMul consumers), raises.
    """
    visited: Set[str] = set()
    cur_outputs = list(start_node.output)
    while True:
        consumers: List[onnx.NodeProto] = []
        seen_ids: Set[int] = set()
        for out_name in cur_outputs:
            for n in consumers_by_tensor.get(out_name, []):
                if id(n) not in seen_ids:
                    seen_ids.add(id(n))
                    consumers.append(n)

        if not consumers:
            raise ValueError(
                f"R3 rotation: forward walk from '{start_node.name}' reached "
                f"a dead end with no MatMul."
            )

        matmuls = [n for n in consumers if n.op_type == "MatMul"]
        if matmuls:
            if len(matmuls) > 1:
                raise ValueError(
                    f"R3 rotation: multiple MatMul consumers reached forward "
                    f"from '{start_node.name}': {[n.name for n in matmuls]}."
                )
            return matmuls[0]

        passthrough = [n for n in consumers if _is_passthrough(n.op_type, n.domain)]
        if len(passthrough) != 1:
            raise ValueError(
                f"R3 rotation: forward walk from '{start_node.name}' is "
                f"ambiguous; expected exactly one pass-through (data-movement) "
                f"op, found {[(n.name, n.op_type) for n in consumers]}."
            )
        nxt = passthrough[0]
        if nxt.name in visited:
            raise ValueError(
                f"R3 rotation: cycle detected at '{nxt.name}' while walking "
                f"forward from '{start_node.name}'."
            )
        visited.add(nxt.name)
        cur_outputs = list(nxt.output)


def _find_q_input_of_qk_matmul(
    qk_matmul_node: onnx.NodeProto,
    concat_node: onnx.NodeProto,
    producer_by_tensor: Dict[str, onnx.NodeProto],
) -> Tuple[int, str]:
    """Return ``(input_idx, tensor_name)`` of the Q-side input of ``qk_matmul_node``.

    The K-side input is whichever traces back (through pass-through ops) to
    ``concat_node``; the Q-side is the other one.
    """
    if len(qk_matmul_node.input) != 2:
        raise ValueError(
            f"R3 rotation: QK^T MatMul '{qk_matmul_node.name}' has "
            f"{len(qk_matmul_node.input)} inputs; expected 2."
        )

    concat_outputs = set(concat_node.output)

    k_idx = None
    for i, inp_name in enumerate(qk_matmul_node.input):
        if _input_traces_back_to(inp_name, concat_outputs, producer_by_tensor):
            if k_idx is not None:
                raise ValueError(
                    f"R3 rotation: both inputs of QK^T MatMul "
                    f"'{qk_matmul_node.name}' trace back to Concat "
                    f"'{concat_node.name}'."
                )
            k_idx = i

    if k_idx is None:
        raise ValueError(
            f"R3 rotation: neither input of QK^T MatMul "
            f"'{qk_matmul_node.name}' traces back to Concat "
            f"'{concat_node.name}'."
        )

    q_idx = 1 - k_idx
    return q_idx, qk_matmul_node.input[q_idx]


def _input_traces_back_to(
    start_tensor_name: str,
    target_outputs: Set[str],
    producer_by_tensor: Dict[str, onnx.NodeProto],
) -> bool:
    """BFS backward from ``start_tensor_name`` through pass-through producers."""
    if start_tensor_name in target_outputs:
        return True
    visited: Set[str] = set()
    queue = deque([start_tensor_name])
    while queue:
        name = queue.popleft()
        if name in visited:
            continue
        visited.add(name)
        if name in target_outputs:
            return True
        producer = producer_by_tensor.get(name)
        if producer is None:
            continue
        if not _is_passthrough(producer.op_type, producer.domain):
            continue
        for inp in producer.input:
            if inp and inp not in visited:
                queue.append(inp)
    return False


__all__ = ["BlockR3Anchors", "find_r3_anchors"]
