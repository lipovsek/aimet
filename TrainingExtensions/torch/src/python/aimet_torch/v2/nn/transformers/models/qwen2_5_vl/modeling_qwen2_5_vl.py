# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Quantized Qwen2.5 VL modules"""

import torch
from aimet_torch.v2.nn.true_quant import QuantizationMixin

try:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
except ImportError as exc:
    raise ImportError(
        "aimet_torch.v2.nn.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl cannot be imported. Please make sure "
        "that you have transformers installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx


# Map Qwen2_5_VLRMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to Qwen2_5_VLRMSNorm
map_torch_types_to_onnx[modeling_qwen2_5_vl.Qwen2RMSNorm] = ["RMSNormalization"]

# Don't simulate quantization on rotary embedding layers
QuantizationMixin.ignore(modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding)
QuantizationMixin.ignore(modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding)


try:
    from transformers.activations import SiLUActivation

    @QuantizationMixin.implements(SiLUActivation)
    class QuantizedSiLU(QuantizationMixin, SiLUActivation):
        """Quantized CustomSiLU"""

        __quant_init__ = QuantizationMixin.__unary__

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
            (input_qtzr,) = self.input_quantizers
            (output_qtzr,) = self.output_quantizers

            if input_qtzr:
                x = input_qtzr(x)

            out = super().forward(x)

            if output_qtzr:
                out = output_qtzr(out)

            return out
except ImportError:
    # Older version of transformers use torch.nn.SiLu directly
    pass


@QuantizationMixin.implements(modeling_qwen2_5_vl.Qwen2RMSNorm)
class QuantizedQwen2RMSNorm(QuantizationMixin, modeling_qwen2_5_vl.Qwen2RMSNorm):
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
