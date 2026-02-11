# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import torchvision
from packaging.version import Version
import pytest
import torch
from torchvision import models

from aimet_torch.model_preparer import prepare_model, _prepare_traced_model
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.v1.quantsim import QuantizationSimModel


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model given dummy input
    :param model: torch model
    :param dummy_input: dummy input to model
    """
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]

    model.eval()
    with torch.no_grad():
        model(*dummy_input)


class TestModelPreparer:
    @pytest.mark.cuda
    def test_inception_v3(self):
        """Verify inception_v3"""
        model = models.inception_v3().eval().cuda()
        prepared_model = prepare_model(model)
        print(prepared_model)
        input_shape = (1, 3, 299, 299)
        dummy_input = torch.randn(*input_shape).cuda()

        # Verify bit-exact outputs.
        assert torch.equal(prepared_model(dummy_input), model(dummy_input))

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify with Quantization workflow.
        quant_sim = QuantizationSimModel(prepared_model, dummy_input=dummy_input)
        quant_sim.compute_encodings(evaluate, dummy_input)
        quant_sim.model(dummy_input)

    @pytest.mark.cuda
    def test_deeplab_v3(self):
        """Verify deeplab_v3"""
        # Set the strict flag to False so that torch.jit.trace can be successful.
        from aimet_torch.meta import connectedgraph

        connectedgraph.jit_trace_args.update({"strict": False})
        if Version(torchvision.__version__) < Version("0.10.2"):
            model = (
                models.segmentation.deeplabv3_resnet50(pretrained_backbone=False)
                .eval()
                .cuda()
            )
        else:
            model = (
                models.segmentation.deeplabv3_resnet50(weights_backbone=None)
                .eval()
                .cuda()
            )
        prepared_model = prepare_model(model)
        print(prepared_model)
        input_shape = (1, 3, 224, 224)
        dummy_input = torch.randn(*input_shape).cuda()

        # Verify bit-exact outputs.
        assert torch.equal(
            prepared_model(dummy_input)["out"], model(dummy_input)["out"]
        )

        # Verify that validator checks pass.
        assert ModelValidator.validate_model(prepared_model, dummy_input)

        # Verify with Quantization workflow.
        quant_sim = QuantizationSimModel(prepared_model, dummy_input=dummy_input)
        quant_sim.compute_encodings(evaluate, dummy_input)
        quant_sim.model(dummy_input)

    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed
        # So adding a dummy test to satisfy pytest
        pass
