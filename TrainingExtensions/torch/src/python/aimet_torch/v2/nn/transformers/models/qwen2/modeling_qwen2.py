# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Quantized Qwen2 modules"""

import torch
from aimet_torch.v2.nn.true_quant import QuantizationMixin

try:
    from transformers.models.qwen2 import modeling_qwen2
except ImportError as exc:
    raise ImportError(
        "aimet_torch.v2.nn.transformers.models.qwen2.modeling_qwen2 cannot be imported. Please make sure "
        "that you have transformers installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx


# Map Qwen2RMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to Qwen2RMSNorm
map_torch_types_to_onnx[modeling_qwen2.Qwen2RMSNorm] = ["RMSNormalization"]

# Don't simulate quantization on rotary embedding layers
QuantizationMixin.ignore(modeling_qwen2.Qwen2RotaryEmbedding)


@QuantizationMixin.implements(modeling_qwen2.Qwen2RMSNorm)
class QuantizedQwen2RMSNorm(QuantizationMixin, modeling_qwen2.Qwen2RMSNorm):
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
