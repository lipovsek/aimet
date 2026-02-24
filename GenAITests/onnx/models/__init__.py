# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI models"""

import warnings

from .llama import Llama_32_ONNX, Llama_32_SHA_ONNX, Llama_32_AIHM_ONNX
from .qwen2 import Qwen_25_ONNX, Qwen_25_AIHM_ONNX
from .qwen2_vl import Qwen_25_VL_ONNX
from .phi3 import Phi_3_ONNX, Phi_3_AIHM_ONNX
from .mistral import Mistral_03_ONNX
from .qwen3 import Qwen_3_ONNX, Qwen_3_SHA_ONNX, Qwen_3_SHA_Conv_ONNX, Qwen_3_AIHM_ONNX

try:
    from .qwen3_vl import Qwen_3_VL_ONNX
except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )
    pass
