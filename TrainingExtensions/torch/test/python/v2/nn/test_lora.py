#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import json
import os
import torch
from torch import nn
import peft.tuners.lora.layer as lora
import tempfile

import aimet_torch.v2 as aimet
from aimet_torch.v2.quantization import affine
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import lora as qlora


class TestQuantizedLinear:
    @pytest.mark.parametrize(
        "model, dummy_input",
        [
            (
                lora.Linear(nn.Linear(10, 10), adapter_name="adapter_0", r=1),
                torch.randn(10, 10),
            ),
            (
                lora.Conv2d(nn.Conv2d(10, 10, 1, 1), adapter_name="adapter_0", r=1),
                torch.randn(10, 10, 1, 1),
            ),
        ],
    )
    def test_quantsim_basics(self, model, dummy_input):
        model = lora.Linear(nn.Linear(10, 10), adapter_name="adapter_0", r=1)
        dummy_input = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input)

        """
        When: Create quantsim with lora.Linear
        Then: 1) lora.Linear should be converted to QuantizedLinear
              2) Mul and Add modules should have input and output quantizers as necessary
              3) All lora adapters (lora_A, B) and base layer should be converted to aimet.nn.QuantizedLinear
        """
        assert isinstance(sim.model, qlora.QuantizedLora)
        assert isinstance(
            sim.model.mul["adapter_0"].input_quantizers[1], affine.QuantizeDequantize
        )
        assert isinstance(
            sim.model.mul["adapter_0"].output_quantizers[0], affine.QuantizeDequantize
        )
        assert isinstance(
            sim.model.add["adapter_0"].output_quantizers[0], affine.QuantizeDequantize
        )

        lora_A = sim.model.lora_A["adapter_0"]
        assert type(lora_A) in [aimet.nn.QuantizedLinear, aimet.nn.QuantizedConv2d]
        assert isinstance(lora_A.param_quantizers["weight"], affine.QuantizeDequantize)
        assert isinstance(lora_A.output_quantizers[0], affine.QuantizeDequantize)

        lora_B = sim.model.lora_B["adapter_0"]
        assert type(lora_B) in [aimet.nn.QuantizedLinear, aimet.nn.QuantizedConv2d]
        assert isinstance(lora_B.param_quantizers["weight"], affine.QuantizeDequantize)
        assert isinstance(lora_B.output_quantizers[0], affine.QuantizeDequantize)

        base_layer = sim.model.base_layer
        assert type(base_layer) in [aimet.nn.QuantizedLinear, aimet.nn.QuantizedConv2d]
        assert isinstance(
            base_layer.param_quantizers["weight"], affine.QuantizeDequantize
        )
        assert isinstance(base_layer.output_quantizers[0], affine.QuantizeDequantize)

        """
        When: compute_encodings
        Then: All quantizers should be initialized
        """
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        for qtzr in sim.model.modules():
            if isinstance(qtzr, QuantizerBase):
                assert qtzr.is_initialized()

        """
        When: Export
        Then: The generated encoding file should contain all entries properly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(tmpdir, "model", dummy_input=dummy_input)
            with open(os.path.join(tmpdir, "model_torch.encodings")) as f:
                encodings = json.load(f)

        expected_schema = {
            "activation_encodings": {
                "base_layer": {"input": {"0": ...}, "output": ...},
                "lora_A.adapter_0": {"input": {"0": ...}, "output": ...},
                "lora_B.adapter_0": {"output": ...},
                "mul.adapter_0": {"input": {"1": ...}, "output": ...},
                "add.adapter_0": {"output": ...},
            },
            "param_encodings": {
                "base_layer.weight": ...,
                "lora_A.adapter_0.weight": ...,
                "lora_B.adapter_0.weight": ...,
            },
        }

        def _assert_same_keys(d: dict, expected: dict):
            assert d.keys() == expected.keys()

            for k in d:
                v1, v2 = d[k], expected[k]
                if isinstance(v2, dict):
                    _assert_same_keys(v1, v2)

        _assert_same_keys(
            encodings["activation_encodings"], expected_schema["activation_encodings"]
        )
        _assert_same_keys(
            encodings["param_encodings"], expected_schema["param_encodings"]
        )

    @pytest.mark.skip(reason="To be discussed")
    def test_update_layer(self):
        """
        When: Add a new lora adapter with "update_layer" API
        Then: The new added adapters should be aimet.nn.QuantizedLinear with
              param and output quantizers instantiated as necessary
        """
        model = lora.Linear(nn.Linear(10, 10), adapter_name="adapter_0", r=1)
        dummy_input = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input)

        sim.model.update_layer("new_adapter", ...)
        new_lora_a = sim.model.lora_A["new_adapter"]
        new_lora_b = sim.model.lora_B["new_adapter"]

        assert isinstance(new_lora_a, aimet.nn.QuantizedLinear)
        assert isinstance(
            new_lora_a.param_quantizers["weight"], affine.QuantizeDequantize
        )
        assert isinstance(new_lora_a.output_quantizers[0], affine.QuantizeDequantize)

        assert isinstance(new_lora_b, aimet.nn.QuantizedLinear)
        assert isinstance(
            new_lora_b.param_quantizers["weight"], affine.QuantizeDequantize
        )
        assert isinstance(new_lora_b.output_quantizers[0], affine.QuantizeDequantize)

        assert isinstance(
            sim.model.mul["new_adapter"].input_quantizers[1], affine.QuantizeDequantize
        )
        assert isinstance(
            sim.model.mul["new_adapter"].output_quantizers[0], affine.QuantizeDequantize
        )
        assert isinstance(
            sim.model.add["new_adapter"].output_quantizers[0], affine.QuantizeDequantize
        )
