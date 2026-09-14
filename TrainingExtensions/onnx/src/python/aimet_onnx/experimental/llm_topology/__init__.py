# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM decoder-stack topology: block boundaries + intra-block structure.

Top-level entry point for describing the structure of an ONNX decoder-stack
model — where the blocks are, and what the q/k/v/o, gate/up/down projections
and dynamic attention MatMuls are inside each block. Technique-agnostic.

Analysis runs on onnx_ir and reports results by ONNX name
(:class:`LlmTopologyByName`). :func:`analyze_llm_topology` additionally
re-attaches a ConnectedGraph and returns the ``Op``-bearing
:class:`LlmTopology`, for consumers that have not migrated yet; that adapter is
transitional and new code should prefer
:func:`analyze_llm_topology_by_name`.
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
