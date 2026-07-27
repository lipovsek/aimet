# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI Torch models"""

# Register the default Torch LLM class
from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

# Register shared adaptations (SHA, SHA_Conv)
from GenAILab.qai_hub_lm import transforms as adaptations  # noqa: F401

# VLM models (registered via the shared VLM backend)
from GenAILab.qai_hub_lm.backends.torch.vlm import *  # noqa: F401, F403

# LLM models that require special handling
try:
    from GenAILab.qai_hub_lm.backends.torch.qwen3_5 import Qwen3_5
except ImportError:
    warnings.warn(
        "Qwen 3.5 is not available. Please upgrade to a later version of transformers to use this model."
    )

# Gemma4 Torch backend (packed-QAT checkpoint support)
try:
    from GenAILab.qai_hub_lm.backends.torch.gemma4 import Gemma4_Torch
except ImportError:
    pass
