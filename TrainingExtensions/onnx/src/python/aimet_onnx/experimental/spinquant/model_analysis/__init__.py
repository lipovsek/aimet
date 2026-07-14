# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Rotation-specific model analysis for SpinQuant.

Holds the analysis each rotation pass needs: R2 attention topology (V/O
identification), R3 attention anchors (Q/K from the KV cache), the VLM visual
merger, and the R1 post-writing-norm precondition check. Technique-agnostic
decoder block detection and role mapping live in ``block_topology``; import
those directly from there.
"""

from aimet_onnx.experimental.spinquant.model_analysis.attention_anchors import (
    BlockR3Anchors,
    find_r3_anchors,
)
from aimet_onnx.experimental.spinquant.model_analysis.attention_topology import (
    BlockAttentionTopology,
    find_attention_topology,
)
from aimet_onnx.experimental.spinquant.model_analysis.norm_detection import (
    find_post_writing_norms,
)
from aimet_onnx.experimental.spinquant.model_analysis.visual_merger import (
    find_merger_linear2,
)

__all__ = [
    "BlockAttentionTopology",
    "BlockR3Anchors",
    "find_attention_topology",
    "find_r3_anchors",
    "find_merger_linear2",
    "find_post_writing_norms",
]
