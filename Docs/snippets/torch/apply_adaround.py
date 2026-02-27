# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
# [setup]
import torch
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from datasets import load_dataset
from evaluate import evaluator

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)
num_batches = 32
data = load_dataset('imagenet-1k', streaming=True, split="train")
data_loader = DataLoader(data, batch_size=num_batches, num_workers = 4)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

def forward_pass(model: torch.nn.Module):
    with torch.no_grad():
        for images, _ in data_loader:
            model(images)

path = './'
filename = 'mobilenet'

# [step_1]
from aimet_torch import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

params = AdaroundParameters(data_loader=data_loader, num_batches=num_batches)

# Returns model with AdaRound-ed weights and their corresponding encodings
adarounded_model = Adaround.apply_adaround(model, dummy_input, params, path=path, filename_prefix=filename)
# [step_2]
sim = QuantizationSimModel(adarounded_model, dummy_input)

# AdaRound optimizes the rounding of weight quantizers only. These values are preserved through load_encodings()
sim.load_encodings(encodings=path + filename, allow_overwrite=False)

# The activation quantizers remain uninitialized and derived through compute_encodings()
sim.compute_encodings(forward_pass)
# [step_3]
evaluator = evaluator("image-classification")
accuracy = evaluator.compute(model_or_pipeline=model, data=data, metric="accuracy")
# [step_4]
import os
import aimet_torch
aimet_torch.onnx.export(
    sim.model,
    dummy_input,
    os.path.join(path, f"quantized_{filename}.onnx"),
    dynamo=False,
    opset_version=21,
)
