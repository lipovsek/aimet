# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI ONNX models"""

import warnings

# Register the default ONNX LLM class
from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX

# Register shared adaptations
from GenAILab.qai_hub_lm import transforms as shared_adaptations  # noqa: F401

# Register ONNX-specific adaptations
from GenAILab.qai_hub_lm.backends.onnx import adaptations  # noqa: F401

# VLM models
from GenAILab.qai_hub_lm.backends.onnx.qwen2_vl import Qwen_25_VL_ONNX

try:
    from GenAILab.qai_hub_lm.backends.onnx.qwen3_vl import Qwen_3_VL_ONNX
except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )

try:
    from GenAILab.qai_hub_lm.backends.onnx.gemma3 import Gemma3_ONNX
except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.backends.onnx.gemma4 import Gemma4_ONNX
except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.backends.onnx.internvl import InternVL_ONNX
except ImportError:
    pass
