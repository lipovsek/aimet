# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Unit tests for Adaround"""

import pytest
import torch

from aimet_torch.common.utils import AimetLogger
import aimet_torch.v2.quantsim as v2
from .models.test_models import TinyModel
from aimet_torch.utils import create_fake_data_loader, CachedDataset
from aimet_torch._base.adaround.activation_sampler import ActivationSampler
from aimet_torch.v2.nn.base import BaseQuantizationMixin

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


@pytest.fixture
def model():
    return TinyModel().eval()


@pytest.fixture
def sim(model):
    sim = v2.QuantizationSimModel(
        model,
        dummy_input=torch.randn(1, 3, 32, 32),
        quant_scheme="tf_enhanced",
        default_param_bw=4,
    )

    for module in sim.model.modules():
        if isinstance(module, BaseQuantizationMixin):
            module._remove_activation_quantizers()

    return sim


class TestAdaroundActivationSampler:
    """
    Adaround unit tests
    """

    def test_activation_sampler_conv(self, sim, model, tmpdir):
        """Test ActivationSampler for a Conv module"""
        dataset_size = 100
        batch_size = 10
        image_size = (3, 32, 32)
        data_loader = create_fake_data_loader(dataset_size, batch_size, image_size)
        possible_batches = dataset_size // batch_size

        def forward_fn(model, inputs):
            inputs, _ = inputs
            model(inputs)

        act_sampler = ActivationSampler(
            model.conv1, sim.model.conv1, model, sim.model, forward_fn
        )
        cached_dataset = CachedDataset(data_loader, possible_batches, tmpdir)
        quant_inp, orig_out = act_sampler.sample_and_place_all_acts_on_cpu(
            cached_dataset
        )

        assert list(quant_inp.shape) == [batch_size * possible_batches, 3, 32, 32]
        assert list(orig_out.shape) == [batch_size * possible_batches, 32, 18, 18]

    def test_activation_sampler_fully_connected_module(self, sim, model, tmpdir):
        """Test ActivationSampler for a fully connected module"""
        dataset_size = 100
        batch_size = 10
        image_size = (3, 32, 32)
        possible_batches = dataset_size // batch_size
        data_loader = create_fake_data_loader(dataset_size, batch_size, image_size)

        def forward_fn(model, inputs):
            inputs, _ = inputs
            model(inputs)

        act_sampler = ActivationSampler(
            model.fc, sim.model.fc, model, sim.model, forward_fn
        )
        cached_dataset = CachedDataset(data_loader, possible_batches, tmpdir)
        quant_inp, orig_out = act_sampler.sample_and_place_all_acts_on_cpu(
            cached_dataset
        )

        assert list(quant_inp.shape) == [batch_size * possible_batches, 36]
        assert list(orig_out.shape) == [batch_size * possible_batches, 12]
