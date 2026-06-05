#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import onnx
import pytest
import os
import torch
from torch import nn
import peft.tuners.lora.layer as lora
import tempfile

import aimet_torch
import aimet_torch.v2 as aimet
from aimet_torch.v2.quantization import affine
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import lora as qlora


class TestQuantizedLinear:
    def test_quantsim_basics(self):
        model = torch.nn.Sequential(
            lora.Linear(nn.Linear(10, 10, bias=False), adapter_name="adapter_0", r=1),
            lora.Conv2d(
                nn.Conv2d(10, 10, 3, bias=False), adapter_name="adapter_0", r=1
            ),
        )
        dummy_input = torch.randn(10, 10, 10)
        sim = QuantizationSimModel(model, dummy_input)

        """
        When: Create quantsim with lora.Linear
        Then: 1) lora.Linear should be converted to QuantizedLinear
              2) Mul and Add modules should have input and output quantizers as necessary
              3) All lora adapters (lora_A, B) and base layer should be converted to aimet.nn.QuantizedLinear
        """
        for qmodule in sim.model:
            assert isinstance(qmodule, qlora.QuantizedLora)
            assert isinstance(
                qmodule.mul["adapter_0"].input_quantizers[1], affine.QuantizeDequantize
            )
            assert isinstance(
                qmodule.mul["adapter_0"].output_quantizers[0], affine.QuantizeDequantize
            )
            assert isinstance(
                qmodule.add["adapter_0"].output_quantizers[0], affine.QuantizeDequantize
            )

            lora_A = qmodule.lora_A["adapter_0"]
            assert type(lora_A) in [aimet.nn.QuantizedLinear, aimet.nn.QuantizedConv2d]
            assert isinstance(
                lora_A.param_quantizers["weight"], affine.QuantizeDequantize
            )
            assert isinstance(lora_A.output_quantizers[0], affine.QuantizeDequantize)

            lora_B = qmodule.lora_B["adapter_0"]
            assert type(lora_B) in [aimet.nn.QuantizedLinear, aimet.nn.QuantizedConv2d]
            assert isinstance(
                lora_B.param_quantizers["weight"], affine.QuantizeDequantize
            )
            assert isinstance(lora_B.output_quantizers[0], affine.QuantizeDequantize)

            base_layer = qmodule.base_layer
            assert type(base_layer) in [
                aimet.nn.QuantizedLinear,
                aimet.nn.QuantizedConv2d,
            ]
            assert isinstance(
                base_layer.param_quantizers["weight"], affine.QuantizeDequantize
            )
            assert isinstance(
                base_layer.output_quantizers[0], affine.QuantizeDequantize
            )

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
        with pytest.raises(
            RuntimeError,
            match="QuantizationSimModel.export does not support exporting QuantizedLora layers.",
        ):
            sim.export(".", "model", dummy_input=dummy_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_qdq.onnx")
            aimet_torch.onnx.export(sim.model, dummy_input, path)
            onnx_model = onnx.load(path)

        producers = {}
        consumers = {}
        for node in onnx_model.graph.node:
            for input in node.input:
                consumers.setdefault(input, []).append(node)
            for output in node.output:
                producers[output] = node

        for node in onnx_model.graph.node:
            if node.op_type in ("QuantizeLinear", "DequantizeLinear", "Transpose"):
                continue
            for input in node.input:
                producer = producers.get(input)
                if producer:
                    assert producer.op_type in ("DequantizeLinear", "Transpose")
            for output in node.output:
                for consumer in consumers.get(output, []):
                    assert consumer.op_type in ("QuantizeLinear",)

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
