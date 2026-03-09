# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""aimet_torch.v2 subpackage"""

from . import experimental
from . import nn
from . import quantization
from . import quantsim
from . import utils
from . import visualization_tools


try:
    from transformers.generation.utils import GenerationMixin
except ImportError:
    GenerationMixin = None


if GenerationMixin:
    # pylint: disable=protected-access
    import functools
    import torch

    _generate = GenerationMixin.generate

    @functools.wraps(GenerationMixin.generate)
    def fast_generate_wrapper(self, *args, **kwargs):
        if not any(
            isinstance(module, quantization.base.QuantizerBase)
            for module in self.modules()
        ):
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
