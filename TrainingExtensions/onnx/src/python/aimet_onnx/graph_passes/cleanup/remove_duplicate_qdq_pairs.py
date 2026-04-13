# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Optimization passes for ONNX QDQ models using onnx-ir.
"""

from typing import Tuple
import numpy as np
import onnx
import onnx_ir
import onnx_ir.passes.common


def _get_const_value(value: onnx_ir.Value | None) -> np.ndarray | None:
    """Get constant value from an onnx-ir Value."""
    if value is None:
        return None
    const = onnx_ir.convenience.get_const_tensor(value)
    if const is None:
        return None
    return const.numpy()


def _qdq_nodes_equal(node1: onnx_ir.Node, node2: onnx_ir.Node) -> bool:
    """
    Check if two QuantizeLinear or DequantizeLinear nodes have equal parameters.

    Compares all attributes (axis, block_size, output_dtype, saturate, etc.) and
    input values (scale, zero_point).
    """
    # Compare all attributes - onnx_ir.Attr has __eq__ defined
    if set(node1.attributes.keys()) != set(node2.attributes.keys()):
        return False
    for key in node1.attributes:
        if node1.attributes[key] != node2.attributes[key]:
            return False

    # Compare scale (input[1])
    scale1 = _get_const_value(node1.inputs[1]) if len(node1.inputs) > 1 else None
    scale2 = _get_const_value(node2.inputs[1]) if len(node2.inputs) > 1 else None
    if scale1 is None and scale2 is None:
        pass
    elif scale1 is None or scale2 is None:
        return False
    elif not np.array_equal(scale1, scale2):
        return False

    # Compare zero_point (input[2])
    zp1 = _get_const_value(node1.inputs[2]) if len(node1.inputs) > 2 else None
    zp2 = _get_const_value(node2.inputs[2]) if len(node2.inputs) > 2 else None
    if zp1 is None and zp2 is None:
        pass
    elif zp1 is None or zp2 is None:
        # One has zero_point, other doesn't - they might still be equal if zp is 0
        if zp1 is not None and not np.all(zp1 == 0):
            return False
        if zp2 is not None and not np.all(zp2 == 0):
            return False
    elif not np.array_equal(zp1, zp2):
        return False

    return True


class RemoveDuplicateQDQPairsPass(onnx_ir.passes.InPlacePass):
    """
    Remove duplicate Q->DQ pairs from an ONNX QDQ model.

    Finds patterns like: input -> Q1 -> DQ1 -> Q2 -> DQ2 -> output
    where Q1/DQ1 have the exact same parameters as Q2/DQ2,
    and replaces them with: input -> Q1 -> DQ2 -> output

    The middle DQ1->Q2 nodes become dead code and should be removed
    by a subsequent RemoveUnusedNodesPass.
    """

    def call(self, model: onnx_ir.Model) -> onnx_ir.passes.PassResult:
        modified = False
        graph = model.graph

        for node in graph:
            # Look for DequantizeLinear nodes (DQ2 in the pattern)
            if node.op_type != "DequantizeLinear":
                continue

            dq2 = node

            # DQ2's input should come from Q2
            if not dq2.inputs or dq2.inputs[0] is None:
                continue

            q2_output = dq2.inputs[0]
            q2 = q2_output.producer()
            if q2 is None or q2.op_type != "QuantizeLinear":
                continue

            # Q2's input should come from DQ1
            if not q2.inputs or q2.inputs[0] is None:
                continue

            dq1_output = q2.inputs[0]
            dq1 = dq1_output.producer()
            if dq1 is None or dq1.op_type != "DequantizeLinear":
                continue

            # DQ1's input should come from Q1
            if not dq1.inputs or dq1.inputs[0] is None:
                continue

            q1_output = dq1.inputs[0]
            q1 = q1_output.producer()
            if q1 is None or q1.op_type != "QuantizeLinear":
                continue

            # Check that DQ1's output is ONLY consumed by Q2 (no other consumers)
            dq1_uses = list(dq1_output.uses())
            if len(dq1_uses) != 1:
                continue

            # Check if Q1/DQ1 params match Q2/DQ2 params
            if not _qdq_nodes_equal(q1, q2):
                continue
            if not _qdq_nodes_equal(dq1, dq2):
                continue

            # Found a duplicate Q->DQ pair!
            # Rewire: DQ2's first input (index 0) becomes Q1's output (bypassing DQ1->Q2)
            dq2.replace_input_with(0, q1_output)
            modified = True

        return onnx_ir.passes.PassResult(model, modified=modified)


def cleanup_qdq_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Run optimization passes on a QDQ model.

    Currently runs:
    1. RemoveDuplicateQDQPairsPass - removes redundant Q->DQ->Q->DQ patterns
    2. RemoveUnusedNodesPass - removes dead code from the graph

    Args:
        model: ONNX ModelProto to optimize

    Returns:
        Optimized ONNX ModelProto
    """
    ir_model = onnx_ir.from_proto(model)

    # List of passes to run - can be extended in the future
    passes = [
        RemoveDuplicateQDQPairsPass(),
        onnx_ir.passes.common.RemoveUnusedNodesPass(),
    ]

    for p in passes:
        p.call(ir_model)

    return onnx_ir.to_proto(ir_model)


def remove_duplicate_qdq_pairs(model: onnx.ModelProto) -> Tuple[int, onnx.ModelProto]:
    """
    Remove duplicate Q->DQ pairs from an ONNX QDQ model.

    Args:
        model: ONNX ModelProto

    Returns:
        Tuple of (removed_count, optimized_model) where removed_count is the
        number of duplicate Q->DQ pairs removed
    """
    q_before = sum(1 for n in model.graph.node if n.op_type == "QuantizeLinear")
    optimized_model = cleanup_qdq_model(model)
    q_after = sum(
        1 for n in optimized_model.graph.node if n.op_type == "QuantizeLinear"
    )
    return q_before - q_after, optimized_model
