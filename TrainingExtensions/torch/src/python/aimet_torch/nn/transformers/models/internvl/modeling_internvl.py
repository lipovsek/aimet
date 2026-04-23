# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantized InternVL modules"""

import torch
from aimet_torch.nn.true_quant import QuantizationMixin

try:
    from transformers.models.internvl import modeling_internvl
except ImportError as exc:
    raise ImportError(
        "aimet_torch.nn.transformers.models.qwen3.modeling_qwen3 cannot be imported. Please make sure "
        "that you have transformers >= 4.51.0 installed in your environment."
    ) from exc

from aimet_torch.onnx_utils import map_torch_types_to_onnx


# Map Qwen2RMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to InternVLVisionRMSNorm
map_torch_types_to_onnx[modeling_internvl.InternVLVisionRMSNorm] = ["RMSNormalization"]


@QuantizationMixin.implements(modeling_internvl.InternVLVisionRMSNorm)
class QuantizedInternVLVisionRMSNorm(
    QuantizationMixin, modeling_internvl.InternVLVisionRMSNorm
):
    """Implement Quantized InternVL RMSNorm"""

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
