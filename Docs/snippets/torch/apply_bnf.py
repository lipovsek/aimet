# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring

# Step 1
import torch
from torchvision.models import mobilenet_v2

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# End of step 1


# Step 2
print("*** Before batch norm folding ***")

print("\nmodel.features[0][0]:")
print(model.features[0][0])

print("\nmodel.features[0][1]:")
print(model.features[0][1])

from aimet_torch.batch_norm_fold import fold_all_batch_norms
fold_all_batch_norms(model, dummy_input.shape, dummy_input=dummy_input)

print("*** After batch norm folding ***")

print("\nmodel.features[0][0]:")
print(model.features[0][0])

print("\nmodel.features[0][1]:")
print(model.features[0][1])

# End of step 2
