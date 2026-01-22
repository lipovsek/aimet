# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch
import torch.nn as nn
from aimet_torch.v1.quantsim import QuantizationSimModel


class TestEntropySchemeStaticGrid:
    """Test Entropy quantization scheme"""

    def test_model_with_entropy_scheme(self):
        """Test entropy scheme"""

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding="same")
                self.conv2 = torch.nn.Conv2d(16, 16, 3, padding="same")

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                return x

        model = Model()
        dummy_input = torch.rand(1, 3, 224, 224)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim1 = QuantizationSimModel(model, dummy_input, quant_scheme="tf")
        sim1.compute_encodings(forward_pass, None)

        sim2 = QuantizationSimModel(model, dummy_input, quant_scheme="tf")

        # Overwrite the quantization scheme
        import aimet_common.libpymo as libpymo
        from aimet_common.defs import MAP_QUANT_SCHEME_TO_PYMO

        MAP_QUANT_SCHEME_TO_PYMO["entropy"] = (
            libpymo.QuantizationMode.QUANTIZATION_ENTROPY
        )

        for _, quant_wrapper in sim2.quant_wrappers():
            for quantizer in quant_wrapper.input_quantizers:
                quantizer.quant_scheme = "entropy"
            for quantizer in quant_wrapper.output_quantizers:
                quantizer.quant_scheme = "entropy"
            for param_quantizer in quant_wrapper.param_quantizers.values():
                param_quantizer.quant_scheme = "entropy"

        sim2.compute_encodings(forward_pass, None)

        # Compare the encoding max between tf and entropy quantization scheme
        assert (
            sim1.model.conv1.output_quantizers[0].encoding.max
            != sim2.model.conv1.output_quantizers[0].encoding.max
        )
