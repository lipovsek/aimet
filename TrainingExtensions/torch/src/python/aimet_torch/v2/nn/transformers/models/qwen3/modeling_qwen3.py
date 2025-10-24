# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Quantized Qwen3 modules"""

import torch
from aimet_torch.v2.nn.true_quant import QuantizationMixin

try:
    from transformers.models.qwen3 import modeling_qwen3
except ImportError as exc:
    raise ImportError(
        "aimet_torch.v2.nn.transformers.models.qwen3.modeling_qwen3 cannot be imported. Please make sure "
        "that you have transformers >= 4.51.0 installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx


# Map Qwen2RMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to Qwen3RMSNorm
map_torch_types_to_onnx[modeling_qwen3.Qwen3RMSNorm] = ["RMSNormalization"]

# Don't simulate quantization on Qwen3RotaryEmbedding layers
QuantizationMixin.ignore(modeling_qwen3.Qwen3RotaryEmbedding)


@QuantizationMixin.implements(modeling_qwen3.Qwen3RMSNorm)
class QuantizedQwen3RMSNorm(QuantizationMixin, modeling_qwen3.Qwen3RMSNorm):
    """Implement Quantized Qwen RMSNorm"""

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
