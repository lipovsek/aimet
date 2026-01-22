# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch
import torch.nn as nn
from aimet_torch.v1.quantsim import QuantizationSimModel


class TestPercentileSchemeStaticGrid:
    """Test Percentile quantization scheme"""

    def test_model_with_percentile_scheme(self):
        """Test pecentile scheme by setting different percentile values"""

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

        sim = QuantizationSimModel(model, dummy_input, quant_scheme="percentile")
        # set same percentile value for all the activation tensors
        sim.set_percentile_value(99.99)
        sim.compute_encodings(forward_pass, None)

        # Assign the same tensor for the outputs and check if the encodings are same
        tensor = torch.rand(1, 3, 224, 224)
        sim.model.conv1.output_quantizers[0].reset_encoding_stats()
        sim.model.conv1.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv1.set_percentile_value(99.999)
        sim.model.conv1.output_quantizers[0].compute_encoding()

        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.set_percentile_value(99.999)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert (
            sim.model.conv1.output_quantizers[0].encoding.max
            == sim.model.conv2.output_quantizers[0].encoding.max
        )
        assert (
            sim.model.conv1.output_quantizers[0].encoding.delta
            == sim.model.conv2.output_quantizers[0].encoding.delta
        )

        # Set different percentile values for each layer and verify that the encoding are not same
        sim.model.conv1.output_quantizers[0].reset_encoding_stats()
        sim.model.conv1.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv1.set_percentile_value(99)
        sim.model.conv1.output_quantizers[0].compute_encoding()

        sim.model.conv2.output_quantizers[0].reset_encoding_stats()
        sim.model.conv2.output_quantizers[0].update_encoding_stats(tensor)
        sim.model.conv2.set_percentile_value(99.999)
        sim.model.conv2.output_quantizers[0].compute_encoding()
        assert (
            sim.model.conv1.output_quantizers[0].encoding.max
            != sim.model.conv2.output_quantizers[0].encoding.max
        )
        assert (
            sim.model.conv1.output_quantizers[0].encoding.delta
            != sim.model.conv2.output_quantizers[0].encoding.delta
        )
