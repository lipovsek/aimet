# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
ONNX graph pass that removes LoRA adapter branches of the form:

    MatMul (down_proj) -> MatMul (up_proj) -> Mul (scaling) -> Add (merge)

from an ONNX model.
"""

import logging
from typing import Dict, List, Sequence, Tuple

import onnx
import onnx_ir

from aimet_onnx.common.onnx._utils import _is_grid_preserving_op


logger = logging.getLogger(__name__)

NodeChain = Tuple[onnx_ir.Node, ...]

LORA_MATMUL_ADAPTER_PATTERN: Tuple[str, ...] = ("MatMul", "MatMul", "Mul", "Add")
LORA_CONV_ADAPTER_PATTERN: Tuple[str, ...] = ("Conv", "Conv", "Mul", "Add")


def _walk_through_grid_preserving_ops(
    node: onnx_ir.Node,
) -> Tuple[onnx_ir.Node, List[onnx_ir.Node]]:
    """
    Starting from `node`, follow the single-consumer chain of grid-preserving
    ops and return the first non-grid-preserving descendant along with the
    list of grid-preserving ops that were traversed to reach it.

    If at any point the chain branches (more than one consumer) or reaches a
    graph output, the traversal stops at that node.
    """
    traversed: List[onnx_ir.Node] = []
    current = node
    while _is_grid_preserving_op(current.op_type, current.domain):
        out = current.outputs[0]
        uses = list(out.uses())
        if len(uses) != 1:
            break
        traversed.append(current)
        current = out.consumers()[0]
    return current, traversed


def _find_op_chain(ir_model: onnx_ir.Model, pattern: Sequence[str]) -> List[NodeChain]:
    """
    Find all linear chains of nodes in the graph matching the given op-type
    pattern, allowing any number of grid-preserving ops between consecutive
    pattern elements.

    A match requires each intermediate tensor (including tensors connecting
    skipped grid-preserving ops) to have exactly one consumer, so we don't
    match chains that branch out mid-way. The connecting tensor may land on
    any input index of the next node (e.g. an `Add` that merges a base branch
    on input 0 and the adapter branch on input 1).

    Grid-preserving ops traversed between pattern elements are included in the
    returned chain at the positions where they were encountered, so the chain
    length may exceed len(pattern).

    Args:
        ir_model: The ONNX IR model to search.
        pattern: Sequence of op-type names describing the chain to match, e.g.
            ("MatMul", "MatMul", "Mul", "Add").

    Returns:
        A list of node tuples representing matched chains.
    """
    if len(pattern) < 1:
        return []

    matches: List[NodeChain] = []

    for start_node in ir_model.graph:
        if start_node.op_type != pattern[0]:
            continue

        chain: List[onnx_ir.Node] = [start_node]
        current = start_node
        matched = True

        for next_op_type in pattern[1:]:
            out = current.outputs[0]
            if len(list(out.uses())) != 1:
                matched = False
                break

            candidate = out.consumers()[0]
            candidate, skipped = _walk_through_grid_preserving_ops(candidate)
            if candidate.op_type != next_op_type:
                matched = False
                break

            chain.extend(skipped)
            chain.append(candidate)
            current = candidate

        if matched:
            matches.append(tuple(chain))

    return matches


def _first_non_grid_preserving_downstream(
    node: onnx_ir.Node,
) -> Tuple[onnx_ir.Node, onnx_ir.Node] | None:
    """
    Walk the single-consumer chain downstream of `node` through grid-preserving
    ops and return the first non-grid-preserving descendant.

    Returns None if the chain branches (more than one consumer), reaches a
    graph output, or never encounters a non-grid-preserving op.
    """
    current = node
    while True:
        out = current.outputs[0]
        consumers = out.consumers()
        if len(consumers) != 1:
            return None
        nxt = consumers[0]
        if not _is_grid_preserving_op(nxt.op_type, nxt.domain):
            return out, nxt
        current = nxt


def _find_adapter_chains(
    ir_model: onnx_ir.Model, attach_points: List[str]
) -> Dict[str, List[NodeChain]]:
    """
    Find all LoRA adapter chains (MatMul or Conv variants) whose terminal `Add`
    op is the first non-grid-preserving descendant of one of the named
    `attach_points`.

    Args:
        ir_model: The ONNX IR model to search.
        attach_points: Node names that the adapter `Add` must attach to.

    Returns:
        A dict mapping each attach-point name to the list of adapter chains
        merging into it (via a chain of `Add` nodes).
    """

    # ------------------------------------------
    # Step 1: Identify the "Add" nodes where the 1st adapter branches merge back into the main graph
    # Strategy: Starting from attach_points, find all "Add" nodes that are reachable via a single-consumer
    # chain of grid-preserving ops
    # ------------------------------------------
    attach_name_set = set(attach_points)
    attach_to_first_add_map: Dict[str, onnx_ir.Node] = {}
    for node in ir_model.graph:
        if node.name not in attach_name_set:
            continue
        _, downstream = _first_non_grid_preserving_downstream(node)
        if downstream is not None and downstream.op_type == "Add":
            attach_to_first_add_map[node.name] = downstream

    # ------------------------------------------
    # Step 2: Identify all LoRA adapter chains
    # Strategy: Find all chains matching the LoRA adapter patterns
    # ------------------------------------------
    all_adapter_chains = _find_op_chain(
        ir_model, LORA_MATMUL_ADAPTER_PATTERN
    ) + _find_op_chain(ir_model, LORA_CONV_ADAPTER_PATTERN)

    # ------------------------------------------
    # Step 3: Create a dictionary of Add nodes in the chain to the full chain for quick lookup
    # Strategy: Create a mapping from the terminal Add node of each chain to the full chain
    # ------------------------------------------
    add_to_chain_map: Dict[onnx_ir.Node, NodeChain] = {
        chain[-1]: chain for chain in all_adapter_chains
    }

    # ------------------------------------------
    # Step 4: Create a map of attach point to the corresponding adapter chains
    # Walk the Add -> Add chain starting at each attach point's first Add, collecting
    # any adapter chain whose terminal Add matches. This handles multiple LoRA
    # branches merging into the same attach point via a chain of Adds.
    # ------------------------------------------
    lora_adapter_chains: Dict[str, List[NodeChain]] = {}
    for attach_name, first_add in attach_to_first_add_map.items():
        chains_for_attach: List[NodeChain] = []
        current = first_add
        while current is not None and current.op_type == "Add":
            if current in add_to_chain_map:
                chains_for_attach.append(add_to_chain_map[current])
            else:
                logger.error(
                    "Add node '%s' reachable from attach point '%s' is not the "
                    "terminal of any LoRA adapter chain; skipping.",
                    current.name,
                    attach_name,
                )

            out = current.outputs[0]
            consumers = out.consumers()
            if len(consumers) != 1:
                break
            current = consumers[0]

        lora_adapter_chains[attach_name] = chains_for_attach

    for attach_name, chains in lora_adapter_chains.items():
        logger.debug(
            "  Attach point '%s' -> %d adapter chain(s):", attach_name, len(chains)
        )
        for chain in chains:
            logger.debug("    - terminal Add: '%s'", chain[-1].name)

    return lora_adapter_chains


def _disconnect_lora_adapter_chains(
    attach_node: onnx_ir.Node,
    chains: List[NodeChain],
) -> None:
    """
    Disconnect the given LoRA adapter chains for one attach point from the graph.

    1. Find the tensor feeding into the first downstream Add from the attach node
       (via `_first_non_grid_preserving_downstream`).
    2. Find consumers of the output tensor from the last Add in the last chain.
    3. Rewire those consumers to read from the tensor in step 1.
    """
    result = _first_non_grid_preserving_downstream(attach_node)
    if result is None:
        return
    feed_tensor, _ = result

    last_add = chains[-1][-1]
    last_add_output = last_add.outputs[0]
    last_add_output.replace_all_uses_with(feed_tensor)


def remove_lora_adapters(
    model: onnx.ModelProto, attach_points: List[str]
) -> onnx.ModelProto:
    """
    Remove LoRA adapter branches from an ONNX model.

    Only chains whose terminal `Add` op is the first non-grid-preserving
    descendant of one of the nodes in `attach_points` (following a
    single-consumer chain of grid-preserving ops) are kept.

    Args:
        model: Input ONNX ModelProto.
        attach_points: Node names that the adapter `Add` must attach to.

    Returns:
        A new ModelProto with LoRA adapter branches removed.
    """
    ir_model = onnx_ir.from_proto(model)

    node_name_to_node: Dict[str, onnx_ir.Node] = {
        node.name: node for node in ir_model.graph
    }

    lora_adapter_chains = _find_adapter_chains(ir_model, attach_points)

    for attach_name, chains in lora_adapter_chains.items():
        if not chains:
            continue
        attach_node = node_name_to_node.get(attach_name)
        if attach_node is None:
            continue
        _disconnect_lora_adapter_chains(attach_node, chains)

    onnx_ir.passes.common.RemoveUnusedNodesPass().call(ir_model)
    onnx_ir.passes.common.TopologicalSortPass().call(ir_model)

    return onnx_ir.to_proto(ir_model)
