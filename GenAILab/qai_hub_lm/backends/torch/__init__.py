# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI Torch models"""

import warnings

# Register the default Torch LLM class
from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

# Register shared adaptations (SHA, SHA_Conv)
from GenAILab.qai_hub_lm import transforms as adaptations  # noqa: F401

# VLM models
from GenAILab.qai_hub_lm.backends.torch.qwen2_vl import Qwen_25_VL_Torch

try:
    from GenAILab.qai_hub_lm.backends.torch.gemma4 import Gemma4_Torch
except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.backends.torch.qwen3_vl import Qwen_3_VL_Torch
except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )
