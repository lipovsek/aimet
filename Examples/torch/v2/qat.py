# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

################################################################################
"""Set random seeds and prepare data loaders"""

import random
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


def imagenet_dataset(split: str) -> Dataset:
    IMAGENET_DIR = os.getenv("IMAGENET_DIR")
    if not IMAGENET_DIR:
        raise RuntimeError(
            "Environment variable 'IMAGENET_DIR' has not been set. "
            "Please set this variable to the path where ImageNet dataset is downloaded "
            "and organized in the following directory structure:\n\n"
            "<IMAGENET_DIR>\n"
            " ├── test\n"
            " ├── train\n"
            " └── val\n"
        )

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return ImageFolder(root=os.path.join(IMAGENET_DIR, split), transform=transform)


test_data_loader = DataLoader(imagenet_dataset("test"), batch_size=128, shuffle=False)
train_data_loader = DataLoader(imagenet_dataset("train"), batch_size=32, shuffle=True)
################################################################################
""" Create W4A8 QuantizationSimModel with ViT """

import aimet_torch
from torchvision.models import vit_b_16

model = vit_b_16(weights="IMAGENET1K_V1").to(device=device).eval()
dummy_input, _ = next(iter(train_data_loader))
dummy_input = dummy_input.to(device=device)
sim = aimet_torch.QuantizationSimModel(
    model,
    dummy_input,
    default_param_bw=4,
    default_output_bw=8,
    in_place=True,
    config_file="htp_v81",
)
with torch.no_grad(), aimet_torch.nn.compute_encodings(sim.model):
    for i, (images, _) in enumerate(train_data_loader):
        if i == 8:
            break
        _ = sim.model(images.to(device=device))
################################################################################
""" Evaluate initial accuracy """


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader):
    from tqdm import tqdm

    top1 = top5 = 0.0
    n_images = 0

    pbar = tqdm(data_loader)
    for images, labels in pbar:
        images = images.to(device=device)
        labels = labels.unsqueeze(-1).to(device=device)
        logits = model(images)

        top1 += (logits.topk(1).indices == labels).sum()
        top5 += (logits.topk(5).indices == labels).sum()
        n_images += images.size(0)

        top1_accuracy = top1 / n_images
        top5_accuracy = top5 / n_images
        pbar.set_description(
            f"Top-1: {top1_accuracy * 100:.2f}%, Top-5: {top5_accuracy * 100:.2f}%"
        )

    top1_accuracy = top1 / n_images
    top5_accuracy = top5 / n_images
    return top1_accuracy, top5_accuracy


from aimet_torch.v2.utils import remove_all_quantizers

with remove_all_quantizers(sim.model):
    top1, top5 = evaluate(sim.model, test_data_loader)
    print("FP Accuracy:")
    print(f"  * Top-1: {top1 * 100:.2f}%")
    print(f"  * Top-5: {top5 * 100:.2f}%")

top1, top5 = evaluate(sim.model, test_data_loader)
print("Fake-quantized Accuracy (before QAT):")
print(f"  * Top-1: {top1 * 100:.2f}%")
print(f"  * Top-5: {top5 * 100:.2f}%")
################################################################################
""" Run QAT and evaluate post-QAT accuracy"""


def train(model: torch.nn.Module, data_loader: DataLoader, n_iter: int):
    from tqdm import tqdm
    from aimet_torch.quantization.affine import AffineQuantizerBase

    optimizer = torch.optim.AdamW(
        # Only train quantization parameters,
        # keeping base model weights unchanged
        params={
            param
            for module in model.modules()
            for param in module.parameters()
            if isinstance(module, AffineQuantizerBase)
        },
        lr=0.001,
    )
    pbar = tqdm(data_loader, total=n_iter)
    for i, (images, labels) in enumerate(pbar):
        if i == n_iter:
            break
        optimizer.zero_grad()
        images = images.to(device=device)
        labels = labels.to(device=device)

        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss:.2f}")


train(sim.model.train(), train_data_loader, n_iter=2000)
top1, top5 = evaluate(sim.model.eval(), test_data_loader)
print("Fake-quantized Accuracy (after QAT):")
print(f"  * Top-1: {top1 * 100:.2f}%")
print(f"  * Top-5: {top5 * 100:.2f}%")
