# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI ONNX models"""

import warnings

# Register the default ONNX LLM class
from .llm import LLM_ONNX

# Register adaptations
from GenAILab.shared.models import adaptations as shared_adaptations
from . import adaptations

# VLM models
from .qwen2_vl import Qwen_25_VL_ONNX

try:
    from .qwen3_vl import Qwen_3_VL_ONNX
except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )
