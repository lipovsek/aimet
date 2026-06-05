# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import tempfile
import os
import torch
from peft import PeftMixedModel
from peft import LoraConfig
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.lora.peft_utils import (
    freeze_base_model_activation_quantizers,
    freeze_base_model_param_quantizers,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        return x


def two_adapter_model():
    model = DummyModel()
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=4,
        bias="none",
        target_modules=["linear"],
    )

    peft_model = PeftMixedModel(model, lora_config)
    peft_model.add_adapter("default_new", lora_config)
    peft_model.set_adapter(["default", "default_new"])
    return peft_model


class TestLoraAdapterPeft:
    def test_freeze_base_model(self):
        model = two_adapter_model()
        dummy_inputs = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)
        print(sim)

        def forward_pass(model, forward_pass_callback=None):
            return model(dummy_inputs)

        sim.compute_encodings(forward_pass, None)
        qc_lora = sim.model.base_model.model.linear

        assert not _is_frozen(qc_lora.base_layer.param_quantizers["weight"])
        freeze_base_model_param_quantizers(sim)
        freeze_base_model_activation_quantizers(sim)

        assert _is_frozen(qc_lora.base_layer.param_quantizers["weight"])
        assert not _is_frozen(qc_lora.lora_A["default"].param_quantizers["weight"])
        assert not _is_frozen(qc_lora.lora_A["default_new"].param_quantizers["weight"])
        assert not _is_frozen(qc_lora.lora_B["default"].param_quantizers["weight"])
        assert not _is_frozen(qc_lora.lora_B["default_new"].param_quantizers["weight"])

        assert _is_frozen(qc_lora.base_layer.output_quantizers[0])
        assert not _is_frozen(qc_lora.lora_A["default"].output_quantizers[0])
        assert not _is_frozen(qc_lora.lora_B["default_new"].output_quantizers[0])

    def test_lora_flow(self):
        model = two_adapter_model()
        dummy_inputs = torch.randn(10, 10)
        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)

        def forward_pass(model, forward_pass_callback=None):
            return model(dummy_inputs)

        sim.compute_encodings(forward_pass, None)

        # Export lora model
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.onnx.export(
                dummy_inputs,
                os.path.join(tmpdir, "model"),
            )


def _is_frozen(quantizer):
    return (
        quantizer._allow_overwrite == False
        and quantizer.min.requires_grad == False
        and quantizer.max.requires_grad == False
    )
