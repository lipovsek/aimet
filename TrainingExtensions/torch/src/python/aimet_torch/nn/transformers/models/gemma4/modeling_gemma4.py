# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantized Gemma4 modules"""

import torch
from aimet_torch.nn.true_quant import (
    QuantizationMixin,
    QuantizedEmbedding,
)

try:
    from transformers.models.gemma4 import modeling_gemma4
except ImportError as exc:
    raise ImportError(
        "aimet_torch.nn.transformers.models.gemma4.modeling_gemma4 cannot be imported. "
        "Please make sure that you have transformers installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx

# Map Gemma4RMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to Gemma4RMSNorm
map_torch_types_to_onnx[modeling_gemma4.Gemma4RMSNorm] = ["RMSNormalization"]

# These modules compute positional encodings or pooling — no learnable params to quantize
QuantizationMixin.ignore(modeling_gemma4.Gemma4VisionRotaryEmbedding)
QuantizationMixin.ignore(modeling_gemma4.Gemma4TextRotaryEmbedding)
QuantizationMixin.ignore(modeling_gemma4.Gemma4VisionPooler)
QuantizationMixin.ignore(modeling_gemma4.Gemma4AudioRelPositionalEncoding)
QuantizationMixin.ignore(modeling_gemma4.Gemma4AudioCausalConv1d)


@QuantizationMixin.implements(modeling_gemma4.Gemma4TextScaledWordEmbedding)
class QuantizedGemma4TextScaledWordEmbedding(
    QuantizedEmbedding, modeling_gemma4.Gemma4TextScaledWordEmbedding
):
    pass


@QuantizationMixin.implements(modeling_gemma4.Gemma4RMSNorm)
class QuantizedGemma4RMSNorm(QuantizationMixin, modeling_gemma4.Gemma4RMSNorm):
    """Quantized Gemma4RMSNorm — weight param only present when with_scale=True."""

    def __quant_init__(self):
        # pylint: disable=useless-parent-delegation
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        if self.with_scale:
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
