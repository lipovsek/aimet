# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantized Qwen3 MoE modules"""

import torch
from aimet_torch.v2.nn.true_quant import QuantizationMixin

try:
    from transformers.models.qwen3_moe import modeling_qwen3_moe
except ImportError as exc:
    raise ImportError(
        "aimet_torch.v2.nn.transformers.models.qwen3_moe.modeling_qwen3_moe cannot be imported. Please make sure "
        "that you have transformers >= 4.51.0 installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx

# Map Qwen3MoeRMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to Qwen3MoeRMSNorm
map_torch_types_to_onnx[modeling_qwen3_moe.Qwen3MoeRMSNorm] = ["RMSNormalization"]

# Don't simulate quantization on Qwen3RotaryEmbedding layers
QuantizationMixin.ignore(modeling_qwen3_moe.Qwen3MoeRotaryEmbedding)


@QuantizationMixin.implements(modeling_qwen3_moe.Qwen3MoeRMSNorm)
class QuantizedQwen3MoeRMSNorm(QuantizationMixin, modeling_qwen3_moe.Qwen3MoeRMSNorm):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states):
        # Quantize input tensors
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret
