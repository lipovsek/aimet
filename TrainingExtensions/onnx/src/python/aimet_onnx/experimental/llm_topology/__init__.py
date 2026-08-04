# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM decoder-stack topology: block boundaries + intra-block structure.

Top-level entry point for describing the structure of an ONNX decoder-stack
model — where the blocks are, and what the q/k/v/o, gate/up/down projections
and dynamic attention MatMuls are inside each block. Technique-agnostic.
"""

from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
)
from aimet_onnx.experimental.llm_topology.layer_roles import (
    LinearRole,
    classify_linear_role,
)
from aimet_onnx.experimental.llm_topology.norm_detection import (
    ActiveNorm,
    find_active_norms,
)
from aimet_onnx.experimental.llm_topology.topology import (
    BlockTopology,
    LinearGroup,
    LlmTopology,
    analyze_llm_topology,
    get_llm_topology,
)

__all__ = [
    "ActiveNorm",
    "BlockTopology",
    "LinearGroup",
    "LinearRole",
    "LlmTopology",
    "analyze_llm_topology",
    "classify_linear_role",
    "find_active_norms",
    "get_decoder_block_boundaries",
    "get_llm_topology",
]
