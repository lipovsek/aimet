# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Model traversal and role detection for SpinQuant.

This subpackage contains pure traversal/detection logic — it produces data
structures describing decoder block boundaries, RMSNorm scale layers, and the
linear-op role map. None of it knows about specific rotation matrices (R1/R2/R3).
"""

from aimet_onnx.experimental.spinquant.model_analysis.attention_topology import (
    BlockAttentionTopology,
    find_attention_topology,
)
from aimet_onnx.experimental.spinquant.model_analysis.block_identifier import (
    DecoderBlockRoleMap,
    DecoderModelRoleMap,
    get_decoder_block_boundaries,
    get_decoder_role_map,
)
from aimet_onnx.experimental.spinquant.model_analysis.norm_detection import (
    ActiveNorm,
    find_active_norms,
    find_post_writing_norms,
)
from aimet_onnx.experimental.spinquant.model_analysis.visual_merger import (
    find_merger_linear2,
)
from aimet_onnx.experimental.spinquant.model_analysis.weight_utils import (
    get_bias_product,
    get_weight_product,
    infer_head_dim,
    infer_hidden_size,
)

__all__ = [
    "ActiveNorm",
    "BlockAttentionTopology",
    "DecoderBlockRoleMap",
    "DecoderModelRoleMap",
    "find_active_norms",
    "find_attention_topology",
    "find_merger_linear2",
    "find_post_writing_norms",
    "get_bias_product",
    "get_decoder_block_boundaries",
    "get_decoder_role_map",
    "get_weight_product",
    "infer_head_dim",
    "infer_hidden_size",
]
