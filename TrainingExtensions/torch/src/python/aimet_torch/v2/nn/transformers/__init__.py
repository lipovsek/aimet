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
_was_oom = set()


@functools.wraps(GenerationMixin.generate)
def fast_generate_wrapper(self, *args, **kwargs):
    from ...quantization.base import QuantizerBase
    from ...utils import _inference_mode

    if not any(isinstance(module, QuantizerBase) for module in self.modules()):
        # No quantizers found, no need to enter inference mode.
        return _generate(self, *args, **kwargs)

    prequantize_parameters = id(self) not in _was_oom

    try:
        with _inference_mode(self, prequantize_parameters=prequantize_parameters):
            return _generate(self, *args, **kwargs)
    except torch.cuda.OutOfMemoryError as e1:
        _was_oom.add(id(self))

        if not prequantize_parameters:
            raise

        # OOM possibly due to parameter prequantization. Retry without prequantization.
        torch.cuda.empty_cache()

        try:
            with _inference_mode(self, prequantize_parameters=False):
                return _generate(self, *args, **kwargs)
        except torch.cuda.OutOfMemoryError as e2:
            # Failed again with cuda OOM. Reraise from previous OOM error
            raise e2 from e1
        except Exception as e2:
            # New failure unrelated to OOM.
            # Supress the original context to avoid misleading traceback
            raise e2 from None


GenerationMixin.generate = fast_generate_wrapper
del functools, fast_generate_wrapper, GenerationMixin
