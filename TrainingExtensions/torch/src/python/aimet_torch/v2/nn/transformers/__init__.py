# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Base directory to hold quantized transformers modules"""

import functools
import torch
from .models import *
from .activations import *

from transformers.generation.utils import GenerationMixin

_generate = GenerationMixin.generate


@functools.wraps(GenerationMixin.generate)
def fast_generate_wrapper(self, *args, **kwargs):
    # pylint: disable=protected-access
    from ...quantization.base import QuantizerBase
    from ... import utils

    if not any(isinstance(module, QuantizerBase) for module in self.modules()):
        # No quantizers found, no need to enter inference mode.
        return _generate(self, *args, **kwargs)

    try:
        with utils._inference_mode(self, prequantize_parameters=True):
            return _generate(self, *args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        # OOM possibly due to parameter prequantization. Retry without prequantization.
        with utils._inference_mode(self, prequantize_parameters=False):
            return _generate(self, *args, **kwargs)


GenerationMixin.generate = fast_generate_wrapper
del functools, fast_generate_wrapper, GenerationMixin
