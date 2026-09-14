# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-ir related utility functions"""

from typing import Optional

import onnx_ir
from aimet_onnx.graph_passes.fusions.ir_utils import *  # pylint: disable=wildcard-import, unused-wildcard-import

#: ONNX QDQ node types. Unlike ``QcQuantizeOp`` these come in pairs, so removing
#: them takes a use-redirect per node rather than a single output->input map.
_ONNX_QDQ_TYPES = frozenset(("QuantizeLinear", "DequantizeLinear"))


def static_tensor(value: Optional[onnx_ir.Value]) -> Optional[onnx_ir.TensorProtocol]:
    """Return the constant tensor behind ``value``, or None if it is dynamic.

    Covers both forms ConnectedGraph reports as ``is_parm``/``is_const``: a graph
    initializer, and the output of a ``Constant`` node (which is what
    ``do_constant_folding`` exports emit for e.g. RMSNorm gammas).
    """
    if value is None:
        return None
    return onnx_ir.convenience.get_const_tensor(value)


def is_static(value: Optional[onnx_ir.Value]) -> bool:
    """Return True if ``value`` is an initializer or a ``Constant`` node output."""
    return static_tensor(value) is not None


def remove_quantizers(model: onnx_ir.Model) -> None:
    """Remove every quantizer node, rewiring consumers back to the source tensor.

    Covers AIMET's ``QcQuantizeOp`` (via :func:`remove_aimet_quantizers`) and ONNX
    ``QuantizeLinear``/``DequantizeLinear`` pairs, so a quantized graph presents
    the same topology — and the same tensor names — as the float graph it was
    built from.

    :param model: Model to strip, mutated in place.
    """
    remove_aimet_quantizers(model)
    _remove_onnx_qdq(model)


def _remove_onnx_qdq(model: onnx_ir.Model) -> None:
    """Collapse ``QuantizeLinear``/``DequantizeLinear`` pass-throughs in place."""
    _remove_passthrough_nodes(
        [node for node in model.graph if node.op_type in _ONNX_QDQ_TYPES]
    )


def remove_aimet_quantizers(model: onnx_ir.Model):
    """Remove AIMET ``QcQuantizeOp`` nodes, rewiring consumers to the source tensor."""
    _remove_passthrough_nodes(
        [node for node in model.graph.all_nodes() if node.op_type == "QcQuantizeOp"]
    )


def _remove_passthrough_nodes(nodes: list) -> None:
    """Delete single-in/single-out ``nodes``, redirecting their uses to input 0.

    A graph output produced by a pass-through must be re-pointed at the source
    ``Value``, not merely renamed to match it. Renaming leaves ``graph.outputs``
    holding the deleted node's Value, which makes the real producer's output an
    unused, non-output tensor — so the whole graph reads as dead code to any
    later IR pass. It survives an immediate ``to_proto`` (which matches tensors
    by name) and nothing else.

    Because the source Value may carry no declared type or shape while the
    pass-through's output does, the annotation is copied across before the
    re-point; ``onnx.checker`` requires a type on every graph output.
    """
    passthroughs = [
        node
        for node in nodes
        if len(node.outputs) == 1 and node.inputs and node.inputs[0] is not None
    ]

    for node in passthroughs:
        source, produced = node.inputs[0], node.outputs[0]
        if produced.is_graph_output():
            if source.type is None:
                source.type = produced.type
            if source.shape is None:
                source.shape = produced.shape
        # Nodes are visited in graph order, so a chain (QuantizeLinear ->
        # DequantizeLinear) has already had its head redirected and `source` is
        # the true origin by the time the tail is processed.
        onnx_ir.convenience.replace_all_uses_with(
            produced, source, replace_graph_outputs=True
        )

    for node in passthroughs:
        # safe=True detaches the node from its inputs' user lists; without it the
        # node keeps counting as a consumer after removal. Remove from the node's
        # own graph so nodes inside a subgraph are handled.
        node.graph.remove(node, safe=True)
