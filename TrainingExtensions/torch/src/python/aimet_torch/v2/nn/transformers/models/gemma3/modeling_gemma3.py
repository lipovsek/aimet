# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
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
