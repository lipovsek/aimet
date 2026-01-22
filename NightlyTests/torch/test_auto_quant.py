# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import json
import os
import tempfile

import pytest
import torch
import unittest
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms, datasets

from tqdm import tqdm
from aimet_torch.v1.auto_quant import AutoQuant as V1AutoQuant
from aimet_torch import utils
from aimet_torch.v1.quantsim import QuantizationSimModel as V1QuantizationSimModel
from aimet_torch.v2.auto_quant import AutoQuant as V2AutoQuant
from aimet_torch.v2.quantsim import QuantizationSimModel as V2QuantizationSimModel

from test_quantize_resnet18 import model_train, _get_data_loader


class _UnlabeledDatasetWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        images, *_ = self._dataset[index]
        return images


class TestAutoQuant:
    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "qsim, autoquant",
        [(V1QuantizationSimModel, V1AutoQuant), (V2QuantizationSimModel, V2AutoQuant)],
    )
    def test_autoquant_resnet18(self, qsim, autoquant):
        # Train the model using tiny imagenet data
        model = models.resnet18(pretrained=False).cuda()
        model_train(model, epochs=2)

        dummy_input = torch.randn((1, 3, 224, 224)).cuda()
        val_data_loader = _get_data_loader().val_loader
        unlabeled_data_loader = DataLoader(
            _UnlabeledDatasetWrapper(val_data_loader.dataset),
            batch_size=val_data_loader.batch_size,
        )

        def eval_callback(model, _):
            correct = 0
            with utils.in_eval_mode(model), torch.no_grad():
                for image, label in tqdm(val_data_loader):
                    image = image.cuda()
                    label = label.cuda()
                    logits = model(image)
                    top1 = logits.topk(k=1).indices
                    correct += (top1 == label.view_as(top1)).sum()
            return int(correct) / len(val_data_loader.dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            autoquant = autoquant(
                model,
                dummy_input,
                unlabeled_data_loader,
                eval_callback,
                results_dir=temp_dir,
            )
            model, acc, encoding_path = autoquant.optimize(allowed_accuracy_drop=0)

            sim = qsim(
                model, default_param_bw=8, default_output_bw=8, dummy_input=dummy_input
            )
            sim.compute_encodings(autoquant.forward_pass_callback, None)
            vanilla_accuracy = eval_callback(sim.model, None)
        assert acc >= vanilla_accuracy

    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed
        # So adding a dummy test to satisfy pytest
        pass
