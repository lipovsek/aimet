# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Quantized Mistral modules"""

import torch
from aimet_torch.nn.true_quant import QuantizationMixin

try:
    from transformers.models.mistral import modeling_mistral
except ImportError as exc:
    raise ImportError(
        "aimet_torch.nn.transformers.models.mistral.modeling_mistral cannot be imported. Please make sure "
        "that you have transformers installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx


# Map MistralRMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to MistralRMSNorm
map_torch_types_to_onnx[modeling_mistral.MistralRMSNorm] = ["RMSNormalization"]

# Don't simulate quantization on rotary embedding layers
QuantizationMixin.ignore(modeling_mistral.MistralRotaryEmbedding)


@QuantizationMixin.implements(modeling_mistral.MistralRMSNorm)
class QuantizedMistralRMSNorm(QuantizationMixin, modeling_mistral.MistralRMSNorm):
    """Implement Quantized Mistral RMS Norm"""

    def __quant_init__(self):
        # pylint: disable=useless-parent-delegation
        super().__quant_init__()

        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret
