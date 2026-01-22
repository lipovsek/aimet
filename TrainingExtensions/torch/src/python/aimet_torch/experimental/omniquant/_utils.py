# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Helping functions for Omniquant."""

from aimet_torch.v2.nn import (
    QuantizedLinear,
    QuantizedLayerNorm,
    QuantizedConv2d,
)

from .let_quantizer import (
    OmqQuantizeDequantize,
    OmqGemmaWeightQuantizer,
)

from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.nn.transformers.models.llama.modeling_llama import (
    QuantizedLlamaRMSNorm,
)
from aimet_torch.v2.nn.transformers.models.gemma3.modeling_gemma3 import (
    QuantizedGemma3RMSNorm,
)
from aimet_torch.v2.nn.transformers.models.qwen2.modeling_qwen2 import (
    QuantizedQwen2RMSNorm,
)
from aimet_torch.v2.nn.transformers.models.mistral.modeling_mistral import (
    QuantizedMistralRMSNorm,
)

import torch
import numpy as np
import contextlib

SUPPORTED_QUANTIZED_MODULES = (
    QuantizedLinear,
    QuantizedLayerNorm,
    QuantizedConv2d,
    QuantizedLlamaRMSNorm,
    QuantizedGemma3RMSNorm,
    QuantizedQwen2RMSNorm,
    QuantizedMistralRMSNorm,
)
LWC_MODULES = (QuantizedLinear, QuantizedConv2d)
OMQ_QUANTIZERS = (OmqQuantizeDequantize, OmqGemmaWeightQuantizer)


# pylint: disable=no-else-return
def _move_to_device(data, device):
    """Move resources from cpu to gpu"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [_move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(_move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    else:
        return data


def get_sqnr(fp_out, qt_out, eps=1e-10):
    """Compute the sqnr for fp and qt blocks"""
    if isinstance(fp_out, torch.Tensor):
        fp_out = fp_out.cpu().detach().numpy()
    if isinstance(qt_out, torch.Tensor):
        qt_out = qt_out.cpu().detach().numpy()
    quant_error = fp_out - qt_out
    exp_noise = (quant_error**2).mean() + eps
    exp_signal = (fp_out**2).mean() + eps
    sqnr = exp_signal / exp_noise
    sqnr_db = 10 * np.log10(sqnr)
    return sqnr_db


# pylint:disable = protected-access
def disable_quantizers_for_omq(sim: QuantizationSimModel) -> contextlib.ExitStack:
    """
    Get context managers to disable quantizers temporarily

    :param sim: QuantizationSimModel object
    :return: List of context managers to disable quantizers
    """
    exit_stack = contextlib.ExitStack()
    for module in sim.model.modules():
        if not isinstance(module, QuantizationMixin):
            continue

        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            exit_stack.enter_context(module._remove_all_quantizers())
        else:
            exit_stack.enter_context(module._remove_activation_quantizers())

    return exit_stack


def freeze_let_optimized_param_quantizers(sim: QuantizationSimModel):
    """Freeze the param quantizers from LET blockwise training"""

    def _freeze(module):
        for param_quantizer in module.param_quantizers.values():
            if param_quantizer:
                param_quantizer._allow_overwrite = False
                param_quantizer.requires_grad_(False)

    for module in sim.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            _freeze(module)


def get_omq_quantizer(qdq=None):
    """Dynamically inherit from QuantizeDequantize base on qdq type."""
    if isinstance(qdq, QuantizeDequantize):
        new_cls = type("DynamicOmq", (OmqQuantizeDequantize, QuantizeDequantize), {})
    else:
        new_cls = type("DynamicOmq", (OmqQuantizeDequantize, torch.nn.Module), {})

    instance = new_cls(qdq)

    return instance


def replace_with_omniquant_weight_quantizers(omniquant_block_list):
    """Replace all the weight quantizers in supported modules with adascale quantizers"""
    for block in omniquant_block_list:
        for layer in block.modules():
            if isinstance(layer, SUPPORTED_QUANTIZED_MODULES):
                if isinstance(layer, LWC_MODULES):
                    layer.param_quantizers["weight"] = get_omq_quantizer(
                        layer.param_quantizers["weight"]
                    )
                elif isinstance(layer, QuantizedGemma3RMSNorm):
                    layer.param_quantizers["weight"] = OmqGemmaWeightQuantizer()
                else:
                    layer.param_quantizers["weight"] = get_omq_quantizer()

                if getattr(layer, "bias", None) is not None:
                    layer.param_quantizers["bias"] = get_omq_quantizer()
