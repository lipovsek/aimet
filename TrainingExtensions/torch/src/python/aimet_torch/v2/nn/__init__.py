# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
import contextlib
import torch
from aimet_torch.v2.deepspeed_utils import _register_zero3_forward_hooks
from .fake_quant import *
from .true_quant import *
from .base import *
from .modules import custom
from . import lora

try:
    from . import transformers
except ImportError:
    transformers = None


@contextlib.contextmanager
def compute_encodings(model: torch.nn.Module):
    """
    Compute encodings of all quantized modules in the model

    .. warning::
        Encodings of the quantizers loaded with :ref:`QuantizationSimModel.load_encodings`
        with ``allow_overwrite=False`` will be kept unchanged.
    """
    with (
        _register_zero3_forward_hooks(model, use_dummy_params=False),
        contextlib.ExitStack() as stack,
    ):
        for module in model.modules():
            if isinstance(module, BaseQuantizationMixin):  # pylint: disable=undefined-variable
                ctx = module.compute_encodings()
                stack.enter_context(ctx)

        yield


def compute_param_encodings(model: torch.nn.Module):
    """
    Compute encodings of all parameter quantizers in the model

    .. warning::
        Encodings of the quantizers loaded with :ref:`QuantizationSimModel.load_encodings`
        with ``allow_overwrite=False`` will be kept unchanged.
    """
    for module in model.modules():
        if isinstance(module, BaseQuantizationMixin):  # pylint: disable=undefined-variable
            module.compute_param_encodings()
