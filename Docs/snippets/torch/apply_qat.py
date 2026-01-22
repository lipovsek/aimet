# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring
# pylint: disable=all

# setup
import itertools
import torch
import torchvision
from tqdm import tqdm
from aimet_torch.batch_norm_fold import fold_all_batch_norms

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torchvision.models.mobilenet_v2(pretrained=True).eval().to(device)

PATH_TO_IMAGENET = ...
data = torchvision.datasets.ImageNet(PATH_TO_IMAGENET, split="train")
data_loader = torch.utils.data.DataLoader(data, batch_size=64)

dummy_input = torch.randn(1, 3, 224, 224, device=device)
fold_all_batch_norms(model, dummy_input.shape)

@torch.no_grad()
def pass_calibration_data(model: torch.nn.Module):
    # Pass N batches of calibration data through the model
    for images, _ in itertools.islice(data_loader, 10):
        _ = model(images.to(device))

@torch.no_grad()
def evaluate(model, data_loader):
    # Basic ImageNet evaluation function
    correct = 0
    for data, labels in tqdm(data_loader):
        data, labels = data.to(device), labels.to(device)
        logits = model(data)
        correct += (logits.argmax(1) == labels).sum().item()
    return correct / len(data_loader.dataset)

# step_1
from aimet_torch import QuantizationSimModel
sim = QuantizationSimModel(model, dummy_input)
sim.compute_encodings(pass_calibration_data)

accuracy = evaluate(sim.model.eval(), data_loader)
print(f"Quantized accuracy (W8A8): {accuracy}")
# step_2
# Training loop can be replaced with any custom training loop
def train(model, data_loader, optimizer, loss_fn, num_epochs):
    for _ in range(num_epochs):
        for data, labels in tqdm(data_loader):
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-5)
train(sim.model.train(), data_loader, optimizer, loss_fn, num_epochs=2)
accuracy = evaluate(sim.model.eval(), data_loader)
print(f"Model accuracy after QAT: {accuracy}")
# step_3
