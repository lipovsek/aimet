# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import copy
import pytest
import aimet_torch.v2.quantsim as v2
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.common.defs import QuantScheme
from aimet_torch.bn_reestimation import reestimate_bn_stats, _get_active_bn_modules

import torch
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(1350)


class Model(torch.nn.Module):
    """
    Model
    """

    def __init__(self):
        super(Model, self).__init__()
        self._bn = torch.nn.BatchNorm2d(3)
        self._conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x: torch.Tensor):
        return self._conv(self._bn(x))


@pytest.fixture
def fp32_model(data_loader):
    model = Model().cpu()
    # Run forward pass to initalize batchnorm statistics
    with torch.no_grad():
        for data in data_loader:
            model(data)
    return model


@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn((1, 3, 8, 8))


@pytest.fixture(scope="session")
def data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([torch.randn_like(dummy_input[0]) for _ in range(1)])
    return DataLoader(dataset)


def test_reestimation_with_fp32_model(fp32_model, data_loader):
    expected_mean = [torch.mean(data, dim=(0, 2, 3)) for data in data_loader]
    expected_mean = sum(expected_mean) / len(data_loader)
    expected_var = [torch.var(data, dim=(0, 2, 3)) for data in data_loader]
    expected_var = sum(expected_var) / len(data_loader)
    _test_reestimation(fp32_model, data_loader, expected_mean, expected_var)


@pytest.mark.parametrize(
    "quant_scheme",
    [
        QuantScheme.post_training_tf,
        QuantScheme.post_training_tf_enhanced,
        QuantScheme.training_range_learning_with_tf_init,
        QuantScheme.training_range_learning_with_tf_enhanced_init,
    ],
)
def test_reestimation_with_quantsim_model(
    fp32_model, dummy_input, quant_scheme, data_loader
):
    sim = v2.QuantizationSimModel(fp32_model, dummy_input, quant_scheme=quant_scheme)
    sim.compute_encodings(lambda model, _: model(dummy_input), None)
    model = sim.model

    def quantize_input(data):
        input_quantizer = model._bn.input_quantizers[0]
        if isinstance(input_quantizer, QuantizerBase):
            return input_quantizer(data)

        assert input_quantizer is None
        return data

    expected_mean = [
        torch.mean(quantize_input(data), dim=(0, 2, 3)) for data in data_loader
    ]
    expected_mean = sum(expected_mean) / len(data_loader)
    expected_var = [
        torch.var(quantize_input(data), dim=(0, 2, 3)) for data in data_loader
    ]
    expected_var = sum(expected_var) / len(data_loader)

    _test_reestimation(model, data_loader, expected_mean, expected_var)


def _test_reestimation(model, data_loader, expected_mean, expected_var):
    old_params = copy.deepcopy(list(model.parameters()))

    mean_orig, var_orig = [
        (bn.running_mean.clone().detach(), bn.running_var.clone().detach())
        for bn in _get_active_bn_modules(model)
    ][0]

    with reestimate_bn_stats(model, data_loader):
        for bn in _get_active_bn_modules(model):
            assert bn.momentum != 1.0

        mean_reestimated, var_reestimated = [
            (bn.running_mean.clone().detach(), bn.running_var.clone().detach())
            for bn in _get_active_bn_modules(model)
        ][0]

        assert torch.allclose(mean_reestimated, expected_mean)
        assert torch.allclose(var_reestimated, expected_var)

    new_params = list(model.parameters())

    # All the model parameters should remain the same
    for old, new in zip(old_params, new_params):
        assert torch.equal(old, new)

    mean_restored, var_restored = [
        (bn.running_mean.clone().detach(), bn.running_var.clone().detach())
        for bn in _get_active_bn_modules(model)
    ][0]

    assert not torch.equal(mean_orig, mean_reestimated)
    assert torch.equal(mean_orig, mean_restored)

    assert not torch.equal(var_orig, var_reestimated)
    assert torch.equal(var_orig, var_restored)
