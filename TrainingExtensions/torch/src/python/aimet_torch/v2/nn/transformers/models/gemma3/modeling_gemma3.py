# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Quantized Gemma3 modules"""

import torch
from aimet_torch.v2.nn.true_quant import QuantizationMixin

try:
    from transformers.models.gemma3 import modeling_gemma3
    from transformers.activations import PytorchGELUTanh
except ImportError as exc:
    raise ImportError(
        "aimet_torch.v2.nn.transformers.models.gemma3.modeling_gemma3 cannot be imported. Please make "
        "sure that you have transformers installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx


# Map Gemma3RMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to Gemma3RMSNorm
map_torch_types_to_onnx[modeling_gemma3.Gemma3RMSNorm] = ["RMSNormalization"]


@QuantizationMixin.implements(modeling_gemma3.Gemma3RMSNorm)
class QuantizedGemma3RMSNorm(QuantizationMixin, modeling_gemma3.Gemma3RMSNorm):
    """Implement Quantized Gemma RMSNorm"""

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


@QuantizationMixin.implements(modeling_gemma3.Gemma3RotaryEmbedding)
class QuantizedQwen2RotaryEmbedding(
    QuantizationMixin, modeling_gemma3.Gemma3RotaryEmbedding
):
    """Implement Quantized Gemma Rotary Embedding"""

    def __quant_init__(self):
        # pylint: disable=useless-parent-delegation
        super().__quant_init__()

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return super().forward(x, position_ids)


@QuantizationMixin.implements(modeling_gemma3.Gemma3TextScaledWordEmbedding)
class QuantizedGemma3TextScaledWordEmbedding(
    QuantizationMixin, modeling_gemma3.Gemma3TextScaledWordEmbedding
):
    """Implement Quantized Gemma Text Scaled Word Embedding"""

    def __quant_init__(self):
        # pylint: disable=useless-parent-delegation
        super().__quant_init__()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return super().forward(input_ids)


@QuantizationMixin.implements(PytorchGELUTanh)
class QuantizedPytorchGELUTanh(QuantizationMixin, PytorchGELUTanh):
    """Implement Quantized Transformers PytorchGELUTanh function"""

    def __quant_init__(self):
        # pylint: disable=useless-parent-delegation
        super().__quant_init__()

        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret
