# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""onnx_ir primitives shared by the LLM decoder-stack analyzers.

Every analyzer in this package works on an *analysis IR model*: a private
:class:`onnx_ir.Model` derived from the caller's ``ModelProto`` by

1. converting the proto into onnx_ir — the caller's proto is never mutated;
2. stripping quantizer nodes, so a ``QuantizationSimModel`` graph presents the
   same topology and the same tensor names as the float graph it was built
   from; and
3. fusing decomposed RMSNorm patterns into single ``RMSNormalization``
   supergroup nodes, so norm detection becomes a node-type lookup instead of a
   multi-op pattern match.

Step 2 is a correctness requirement, not an optimization. A sim graph
interleaves ``QcQuantizeOp`` between every producer and consumer, which
(a) prevents the RMSNorm fusion from matching at all, (b) hides every static
weight behind a quantizer, so no linear looks weighted, and (c) renames every
*consumed* tensor from ``T`` to ``T_updated`` while leaving ``T`` on the
producer. Callers such as adascale pass ``sim.model.model`` and then feed the
returned tensor names back to the *quantizer-free* float graph, so the
un-suffixed names are the contract.

Unlike ConnectedGraph ``Product``\\ s, an :class:`onnx_ir.Value` already knows
its producer and its consumers, so the analyzers here need no producer/consumer
side tables — only topological indices, which node objects do not carry.
"""

from typing import Dict, List, Optional, Tuple

import onnx_ir

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.graph_passes.fusions import fuse_supergroups
from aimet_onnx.graph_passes.fusions.ir_utils import is_fused_supergroup
from aimet_onnx.ir_utils import is_static, remove_quantizers
from aimet_onnx.utils import ModelProto

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LlmTopology)

#: Weighted-projection node types.
LINEAR_TYPES = frozenset(("MatMul", "Gemm", "Conv"))
#: Token-embedding lookup node types.
EMBEDDING_TYPES = frozenset(("Gather",))
SOFTMAX_TYPES = frozenset(("Softmax",))

#: Node types that only reshape/reformat activations without changing the
#: mathematical relationship between a norm's gamma and a downstream linear
#: weight, so a read-group walk crosses them transparently.
GRID_PRESERVING_TYPES = frozenset(
    (
        "Unsqueeze",
        "Squeeze",
        "Transpose",
        "Reshape",
        "Flatten",
        "Cast",
    )
)

#: Fused norm op type produced by the ``RMSNormalization`` supergroup fusion.
RMS_NORM_TYPE = "RMSNormalization"

#: Input index carrying the weight of a MatMul / Gemm / Conv. The same index for
#: all three by ONNX convention, and the same value ConnectedGraph uses.
WEIGHT_INDEX = 1


def build_analysis_ir(model: ModelProto) -> onnx_ir.Model:
    """Return a private, quantizer-free, RMSNorm-fused onnx_ir view of ``model``.

    ``model`` is not mutated: :func:`onnx_ir.from_proto` builds a separate
    object graph, and every step below touches only that copy. Tensor *data* is
    shared by reference (no weight copies), and nothing here rewrites weights.

    :param model: ONNX ModelProto to analyze. May be a float export or a
        ``QuantizationSimModel`` graph.
    :return: The analysis IR model, topologically sorted.
    """
    ir_model = onnx_ir.from_proto(model)

    # Sorted up front so every index built below is a topological index, and
    # again after fusion because the rewriter splices in replacement nodes.
    ir_model.graph.sort()
    remove_quantizers(ir_model)
    fuse_supergroups(ir_model, patterns=[RMS_NORM_TYPE])
    ir_model.graph.sort()
    return ir_model


def topological_index(ir_model: onnx_ir.Model) -> Dict[onnx_ir.Node, int]:
    """Return ``{node: topological index}`` over the main graph.

    Keyed by the node itself: ``onnx_ir.Node`` hashes by identity, and a node
    object is a stabler key than its name.
    """
    return {node: index for index, node in enumerate(ir_model.graph)}


def tensor_to_first_consumer_index(
    ir_model: onnx_ir.Model,
    topo_index: Dict[onnx_ir.Node, int],
) -> Dict[str, int]:
    """Return ``{tensor name: topological index}`` over first ``inputs[0]`` consumers.

    NOTE: Assumes a decoder block's norm is the first op consuming the residual
      edge. That edge also feeds a later residual ``Add``; ``setdefault`` keeps
      the norm because it precedes the Add in topological order (true for
      pre-norm decoders).
    """
    tensor_to_index: Dict[str, int] = {}
    for node in ir_model.graph:
        if node.inputs and node.inputs[0] is not None and node.inputs[0].name:
            tensor_to_index.setdefault(node.inputs[0].name, topo_index[node])
    return tensor_to_index


def node_by_name(ir_model: onnx_ir.Model) -> Dict[str, onnx_ir.Node]:
    """Return ``{node name: node}`` over the main graph.

    The topology reports nodes by name, so this is how a reported name is turned
    back into its IR node. Names are required (see :func:`node_name`).
    """
    return {node_name(node): node for node in ir_model.graph}


def node_by_output_tensor(ir_model: onnx_ir.Model) -> Dict[str, onnx_ir.Node]:
    """Return ``{output tensor name: producing node}`` over the main graph."""
    return {
        output.name: node
        for node in ir_model.graph
        for output in node.outputs
        if output.name
    }


def get_weight_value(node: onnx_ir.Node) -> Tuple[Optional[onnx_ir.Value], bool]:
    """Return ``(weight_value, is_transposed)`` for a MatMul/Gemm/Conv node.

    Handles two patterns:

    * Direct:   W (static) -> MatMul/Gemm/Conv
    * Indirect: W (static) -> Transpose -> MatMul

    Only :data:`WEIGHT_INDEX` is considered, matching the ONNX convention for all
    three op types (and ConnectedGraph's own ``WEIGHT_INDEX``). Scanning every
    input instead would report a ``Gemm``'s static bias, or a constant left-hand
    operand, as the weight.

    :param node: A MatMul, Gemm, or Conv node.
    :return: ``(weight_value, is_transposed)``. ``weight_value`` is None when the
        node has no static weight (e.g. a dynamic attention MatMul).
        ``is_transposed`` is True when the stored tensor is ``[out, in]`` —
        either a ``Gemm`` with ``transB=1``, or a weight reaching a MatMul
        through a ``Transpose``.
    """
    if len(node.inputs) <= WEIGHT_INDEX:
        return None, False
    weight = node.inputs[WEIGHT_INDEX]

    if is_static(weight):
        return weight, _has_transposed_b(node)

    # W -> Transpose -> MatMul. The Transpose lands on WEIGHT_INDEX as well, so
    # the pre-transpose tensor is what carries the values. Restricted to
    # MatMul/Gemm: a Conv weight is [out, in, *kernel], for which "transposed"
    # is not the [out, in] layout the flag denotes.
    if node.op_type not in ("MatMul", "Gemm"):
        return None, False
    producer = weight.producer() if weight is not None else None
    if producer is not None and producer.op_type == "Transpose":
        for transpose_inp in producer.inputs:
            if is_static(transpose_inp):
                return transpose_inp, True
    return None, False


def _has_transposed_b(node: onnx_ir.Node) -> bool:
    """Return True for a ``Gemm`` with ``transB=1`` (stored weight is ``[out, in]``)."""
    if node.op_type != "Gemm":
        return False
    attr = node.attributes.get("transB")
    return bool(attr.as_int()) if attr is not None else False


def is_weighted_linear(node: onnx_ir.Node) -> bool:
    """Return True if ``node`` is a MatMul/Gemm/Conv with a static weight."""
    return node.op_type in LINEAR_TYPES and get_weight_value(node)[0] is not None


def is_dynamic_matmul(node: onnx_ir.Node) -> bool:
    """Return True if ``node`` is a MatMul with no static weight (both inputs dynamic)."""
    return node.op_type == "MatMul" and get_weight_value(node)[0] is None


def is_rms_norm(node: onnx_ir.Node) -> bool:
    """Return True if ``node`` is a fused AIMET ``RMSNormalization`` supergroup call."""
    return node.op_type == RMS_NORM_TYPE and is_fused_supergroup(node)


def node_name(node: onnx_ir.Node) -> str:
    """Return ``node``'s name, which every node AIMET analyzes is required to have.

    :raises ValueError: If ``node`` is unnamed. Exporters name their nodes and
        ``quantsim._fill_missing_node_names`` fills in any that are missing, so an
        unnamed node here means the graph never went through either — better to
        say so than to invent a placeholder that cannot be resolved later.
    """
    if not node.name:
        raise ValueError(
            f"Node of type '{node.op_type}' (outputs "
            f"{[out.name for out in node.outputs]}) has no name. LLM topology "
            "analysis identifies nodes by name; run the model through an exporter "
            "that names its nodes."
        )
    return node.name


def node_names(nodes: List[onnx_ir.Node]) -> List[str]:
    """Return the names of ``nodes``, in order."""
    return [node_name(node) for node in nodes]


def sorted_by_topology(
    nodes: List[onnx_ir.Node],
    topo_index: Dict[onnx_ir.Node, int],
) -> List[onnx_ir.Node]:
    """Return ``nodes`` in topological order.

    The graph walks below use a LIFO worklist, whose completion order depends on
    branch shape, so their results need re-sorting even though the graph itself is
    sorted. This makes every reported group deterministic and its reading order
    match the graph.
    """
    return sorted(nodes, key=lambda node: topo_index.get(node, -1))
