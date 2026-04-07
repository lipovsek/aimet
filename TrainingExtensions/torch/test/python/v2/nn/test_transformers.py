# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from transformers.models.llama import modeling_llama
from transformers.models.phi3 import modeling_phi3
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.mistral import modeling_mistral
from transformers.models.qwen3_5 import modeling_qwen3_5
import aimet_torch


@pytest.mark.parametrize(
    "rmsnorm_cls",
    [
        modeling_llama.LlamaRMSNorm,
        modeling_phi3.Phi3RMSNorm,
        modeling_qwen2.Qwen2RMSNorm,
        modeling_qwen3.Qwen3RMSNorm,
        modeling_gemma3.Gemma3RMSNorm,
        modeling_mistral.MistralRMSNorm,
        modeling_qwen3_5.Qwen3_5RMSNorm,
    ],
)
def test_rmsnorm_quantsim_config(rmsnorm_cls):
    """
    When: Create quantsim with well-known RMSNorm classes with HTP v81 config file
    Then: RMSNorm weights should be quantized asymmetrically
    """
    rmsnorm = rmsnorm_cls(100)
    model = torch.nn.Sequential(rmsnorm)
    x = torch.randn(1, 100, 100)
    sim = aimet_torch.QuantizationSimModel(model, x, config_file="htp_v81")
    assert not sim.model[0].param_quantizers["weight"].symmetric


@pytest.mark.parametrize(
    "gated_rmsnorm_cls",
    [
        modeling_qwen3_5.Qwen3_5RMSNormGated,
    ],
)
def test_gated_rmsnorm_quantsim_config(gated_rmsnorm_cls):
    """
    When: Create quantsim with well-known RMSNorm classes with HTP v81 config file
    Then: RMSNorm weights should be quantized asymmetrically
    """

    class Wrapper(torch.nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.norm = norm

        def forward(self, x, y):
            return self.norm(x, y)

    rmsnorm = gated_rmsnorm_cls(100)
    model = Wrapper(rmsnorm)
    x = torch.randn(1, 100, 100)
    y = torch.randn(1, 100)
    sim = aimet_torch.QuantizationSimModel(model, (x, y), config_file="htp_v81")
    assert not sim.model.norm.param_quantizers["weight"].symmetric
