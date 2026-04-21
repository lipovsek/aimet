# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import copy
import numpy as np
import torch
import torch.nn as nn

from aimet_torch.common.defs import QuantScheme
import aimet_torch.bias_correction
import aimet_torch.layer_selector
from aimet_torch import bias_correction
from aimet_torch.quantsim import QuantParams
from aimet_torch import batch_norm_fold
from models.mobilenet import MobileNetV2
from models.imagenet_dataloader import ImageNetDataLoader


def evaluate(model, early_stopping_iterations, use_cuda):
    """
    :param model: model to be evaluated
    :param early_stopping_iterations: if None, data loader will iterate over entire validation data
    :return: dummy ouput
    """
    random_input = torch.rand(1, 3, 224, 224)

    return model(random_input)


class TestBiasCorrection:
    @pytest.mark.cuda
    def test_bias_correction_empirical(self):
        torch.manual_seed(10)
        model = MobileNetV2().to(torch.device("cpu"))
        model.eval()
        batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))
        model_copy = copy.deepcopy(model)
        model.eval()
        model_copy.eval()

        image_dir = "./data/tiny-imagenet-200"
        image_size = 224
        batch_size = 1
        num_workers = 1

        data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
        params = QuantParams(
            weight_bw=4,
            act_bw=4,
            round_mode="nearest",
            quant_scheme=QuantScheme.post_training_tf,
        )
        bias_correction.correct_bias(
            model.to(device="cuda"),
            params,
            1,
            data_loader.train_loader,
            1,
            layers_to_ignore=[model.features[0][0]],
        )

        assert np.allclose(
            model.features[0][0].bias.detach().cpu().numpy(),
            model_copy.features[0][0].bias.detach().cpu().numpy(),
        )

        assert not np.allclose(
            model.features[1].conv[0].bias.detach().cpu().numpy(),
            model_copy.features[1].conv[0].bias.detach().cpu().numpy(),
        )

        # To check if wrappers got removed
        assert isinstance(model.features[11].conv[0], nn.Conv2d)

    @pytest.mark.cuda
    def test_bias_correction_hybrid(self):
        torch.manual_seed(10)

        model = MobileNetV2().to(torch.device("cpu"))
        model.eval()
        module_prop_list = aimet_torch.bias_correction.find_all_conv_bn_with_activation(
            model, dummy_input=torch.rand((1, 3, 224, 224))
        )
        batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))
        model_copy = copy.deepcopy(model)
        model.eval()
        model_copy.eval()

        image_dir = "./data/tiny-imagenet-200"
        image_size = 224
        batch_size = 1
        num_workers = 1

        data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
        params = QuantParams(
            weight_bw=4,
            act_bw=4,
            round_mode="nearest",
            quant_scheme=QuantScheme.post_training_tf,
        )

        bias_correction.correct_bias(
            model.to(device="cuda"),
            params,
            1,
            data_loader.train_loader,
            1,
            module_prop_list,
            False,
        )

        assert np.allclose(
            model.features[0][0].bias.detach().cpu().numpy(),
            model_copy.features[0][0].bias.detach().cpu().numpy(),
        )

        assert not np.allclose(
            model.features[1].conv[0].bias.detach().cpu().numpy(),
            model_copy.features[1].conv[0].bias.detach().cpu().numpy(),
        )

        # To check if wrappers got removed
        assert isinstance(model.features[11].conv[0], nn.Conv2d)

    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed
        # So adding a dummy test to satisfy pytest
        pass
