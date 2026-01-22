# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme
from aimet_torch.v1.qc_quantize_op import StaticGridQuantWrapper
from aimet_torch.v1.adaround.adaround_wrapper import AdaroundWrapper


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: torch.nn.Linear(12, 8),
        lambda: torch.nn.Conv2d(12, 8, 3),
        lambda: torch.nn.ConvTranspose2d(12, 8, 3),
    ],
)
def test_adaround_tensor_quantizer(module_factory):
    """Test the Adarounding of a Tensor"""
    nearest_encoding = libpymo.TfEncoding()
    nearest_encoding.bw = 4
    nearest_encoding.max = 10.0
    nearest_encoding.min = 0.19699306
    nearest_encoding.offset = -127.0
    nearest_encoding.delta = 0.001551126479

    module = module_factory()
    weight_tensor = torch.randn(module.weight.shape)
    wrapper = StaticGridQuantWrapper(
        module,
        weight_bw=4,
        activation_bw=4,
        round_mode="nearest",
        quant_scheme=QuantScheme.post_training_tf_enhanced,
    )
    wrapper.param_quantizers["weight"].encoding = nearest_encoding
    ada_wrapper = AdaroundWrapper(wrapper)
    ada_quantized = ada_wrapper.apply_adaround(weight_tensor)
    assert not torch.equal(weight_tensor, ada_quantized)
