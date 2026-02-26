# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI Torch models"""

import warnings

# Register the default Torch LLM class
from .llm import LLM_Torch

# Register shared adaptations (SHA, SHA_Conv)
from GenAITests.shared.models import adaptations

# VLM models
from .qwen2_vl import Qwen_25_VL_Torch

try:
    from .qwen3_vl import Qwen_3_VL_Torch
except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )
