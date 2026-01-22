# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
# Step 1
import torch
from torchvision.models import mobilenet_v2

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)
print(model)
input_shape = (1, 3, 224, 224)

print('*** Before cross-layer equalization ***')

print('\nmodel.features[1].conv[0][0]')
print(model.features[1].conv[0][0])

print('\nmodel.features[1].conv[1]')
print(model.features[1].conv[1])

print('\nmodel.features[1].conv[0][0]')
print(model.features[1].conv[0][0].weight)

print('\nmodel.features[1].conv[1]')
print(model.features[1].conv[1].weight)


# [step_1]
from aimet_torch.cross_layer_equalization import equalize_model

# Performs BatchNorm folding, cross layer scaling and high bias folding
equalize_model(model, input_shape)

print('*** After cross-layer equalization ***')

print('\nmodel.features[1].conv[0][0]')
print(model.features[1].conv[0][0].weight)

print('\nmodel.features[1].conv[1]')
print(model.features[1].conv[1].weight)
