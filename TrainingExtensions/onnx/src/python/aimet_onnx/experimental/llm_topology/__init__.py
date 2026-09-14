# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM decoder-stack topology: block boundaries + intra-block structure.

Top-level entry point for describing the structure of an ONNX decoder-stack
model — where the blocks are, and what the q/k/v/o, gate/up/down projections
and dynamic attention MatMuls are inside each block. Technique-agnostic.

Analysis runs on onnx_ir and reports results by ONNX name
(:class:`LlmTopologyByName`), which is what new code should consume — via
:func:`analyze_llm_topology_by_name`.

Two adapters re-attach a graph to a name-based topology, for consumers that want
handles rather than names:

* :mod:`~.ir_adapter` — returns ``onnx_ir.Node`` / ``onnx_ir.Value`` objects. The
  one to use: an IR value knows its own producer and consumers, and the graph
  stays valid across node insertions. Import it directly; its dataclasses share
  their names with the ConnectedGraph ones below, so they are deliberately not
  re-exported here.
* :mod:`~.cg_adapter` — returns ConnectedGraph ``Op`` / ``Product`` objects, and
  is what the names re-exported from this package refer to. Transitional; it is
  expected to be deleted once its remaining consumers migrate.
"""

from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
)
from aimet_onnx.experimental.llm_topology.cg_adapter import (
    ActiveNorm,
    BlockTopology,
    LinearGroup,
    LlmTopology,
    resolve_active_norms,
    resolve_topology,
)
from aimet_onnx.experimental.llm_topology.layer_roles import (
    LinearRole,
    classify_linear_role,
)
from aimet_onnx.experimental.llm_topology.norm_detection import (
    ActiveNormByName,
    find_active_norms,
)
from aimet_onnx.experimental.llm_topology.topology import (
    analyze_llm_topology,
    analyze_llm_topology_by_name,
    get_llm_topology,
)
from aimet_onnx.experimental.llm_topology.topology_by_name import (
    BlockTopologyByName,
    LinearGroupByName,
    LlmTopologyByName,
)

__all__ = [
    "ActiveNorm",
    "ActiveNormByName",
    "BlockTopology",
    "BlockTopologyByName",
    "LinearGroup",
    "LinearGroupByName",
    "LinearRole",
    "LlmTopology",
    "LlmTopologyByName",
    "analyze_llm_topology",
    "analyze_llm_topology_by_name",
    "classify_linear_role",
    "find_active_norms",
    "get_decoder_block_boundaries",
    "get_llm_topology",
    "resolve_active_norms",
    "resolve_topology",
]
