# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantized Qwen3.5 modules"""

import torch
from aimet_torch.nn.true_quant import QuantizationMixin

try:
    from transformers.models.qwen3_5 import modeling_qwen3_5
except ImportError as exc:
    raise ImportError(
        "aimet_torch.nn.transformers.models.qwen3_5.modeling_qwen3_5 cannot be imported. Please make sure "
        "that you have transformers >= 4.57.0 installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx

# Map Qwen3_5RMSNorm and Qwen3_5RMSNormGated to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to both variants
map_torch_types_to_onnx[modeling_qwen3_5.Qwen3_5RMSNorm] = ["RMSNormalization"]
map_torch_types_to_onnx[modeling_qwen3_5.Qwen3_5RMSNormGated] = ["RMSNormalization"]

# Don't simulate quantization on rotary embedding layers
QuantizationMixin.ignore(modeling_qwen3_5.Qwen3_5TextRotaryEmbedding)
QuantizationMixin.ignore(modeling_qwen3_5.Qwen3_5VisionRotaryEmbedding)


@QuantizationMixin.implements(modeling_qwen3_5.Qwen3_5RMSNorm)
class QuantizedQwen3_5RMSNorm(QuantizationMixin, modeling_qwen3_5.Qwen3_5RMSNorm):
    """Quantized Qwen3.5 RMSNorm"""

    def __quant_init__(self):
        super().__quant_init__()

        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


@QuantizationMixin.implements(modeling_qwen3_5.Qwen3_5RMSNormGated)
class QuantizedQwen3_5RMSNormGated(
    QuantizationMixin, modeling_qwen3_5.Qwen3_5RMSNormGated
):
    """Quantized Qwen3.5 Gated RMSNorm"""

    def __quant_init__(self):
        super().__quant_init__()

        self.input_quantizers = torch.nn.ModuleList([None, None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states: torch.Tensor, gate=None) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)
        if gate is not None and self.input_quantizers[1]:
            gate = self.input_quantizers[1](gate)

        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states, gate=gate)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret
