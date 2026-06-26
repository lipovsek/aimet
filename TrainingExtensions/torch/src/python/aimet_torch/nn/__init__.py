# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
import contextlib
import torch
from aimet_torch.deepspeed_utils import _register_zero3_forward_hooks
from .fake_quant import *
from .true_quant import *
from .base import *
from .modules import custom
from . import lora
from ..quantization.base import QuantizerBase

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
    from ..utils import remove_activation_quantizers

    activation_quantizers = set()
    param_quantizers = set()
    standalone_quantizers = set()
    qmodules = set()

    for module in model.modules():
        if isinstance(module, BaseQuantizationMixin):
            activation_quantizers |= set(
                itertools.chain(
                    module.output_quantizers.children(),
                    module.input_quantizers.children(),
                )
            )
            param_quantizers |= set(module.param_quantizers.children())
            qmodules.add(module)

    for qtzr in model.modules():
        if (
            isinstance(qtzr, QuantizerBase)
            and qtzr not in param_quantizers
            and qtzr not in activation_quantizers
        ):
            standalone_quantizers.add(qtzr)

    with (
        _register_zero3_forward_hooks(model, use_dummy_params=False),
        contextlib.ExitStack() as stack,
    ):
        for qmodule in qmodules:
            # Enter compute_encodings with activation quantizers temporarily
            # removed. Activation quantizers will enter compute_encodings in
            # the subsequent loop separately
            with remove_activation_quantizers(qmodule):
                ctx = qmodule.compute_encodings()
                stack.enter_context(ctx)

        # Some qmoules such as QuantizedLinear recalibrate the weight encoding
        # as they exit compute_encodings to prevent int32 bias overflow.
        # Since this recalibration process requires the input scale to be
        # computed first, we need to make sure that activation quantizers'
        # compute_encodings finishes earlier than that of qmodules. Therefore,
        # enter activation quantizer's compute_encodings at the top of ExitStack
        for qtzr in itertools.chain(
            activation_quantizers,
            standalone_quantizers,
        ):
            if qtzr.is_overwrite_allowed():
                passthrough = qtzr in activation_quantizers
                ctx = qtzr._compute_encodings(passthrough=passthrough)  # pylint: disable=protected-access
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
