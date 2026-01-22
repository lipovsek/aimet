#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""contains unit tests to validate transformer quantization support"""

import unittest
import torch

from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.transformers.utils import get_quantizable_pt_transformer_model


class TestQuantizationSimTransformers(unittest.TestCase):
    def test_word_langauge_model(self):
        from transformer_models.word_language_model import TransformerModel

        n_layers = 2
        model = TransformerModel(33278, 200, 2, 200, n_layers)

        model.eval()
        get_quantizable_pt_transformer_model(model)

        # create quantsim object on updated model
        dummy_input = torch.randint(33278, size=(35, 20))
        sim = QuantizationSimModel(model, dummy_input)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim.compute_encodings(forward_pass, None)

        for i in range(n_layers):
            # validate MHA layers have quantizers
            self.assertTrue(
                sim.model.transformer_encoder.layers[i]
                .self_attn.linear_Q.output_quantizers[0]
                .encoding
            )
            self.assertTrue(
                sim.model.transformer_encoder.layers[i]
                .self_attn.linear_K.output_quantizers[0]
                .encoding
            )
            self.assertTrue(
                sim.model.transformer_encoder.layers[i]
                .self_attn.linear_V.output_quantizers[0]
                .encoding
            )
            self.assertTrue(
                sim.model.transformer_encoder.layers[i]
                .self_attn.matmul_1.output_quantizers[0]
                .encoding
            )
            self.assertTrue(
                sim.model.transformer_encoder.layers[i]
                .self_attn.matmul_2.output_quantizers[0]
                .encoding
            )
            self.assertTrue(
                sim.model.transformer_encoder.layers[i]
                .self_attn.softmax.output_quantizers[0]
                .encoding
            )
        del sim
