# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
PyTorch Model Evaluation Module

Provides utilities for evaluating PyTorch models using native PyTorch execution.
This module complements eval_onnx.py by supporting direct PyTorch model evaluation
without ONNX conversion.

Key functionality:
- Native PyTorch model evaluation on classification datasets
- Dataset resolution from model metadata
- Consistent interface with ONNX evaluation
- Device-aware preprocessing handling for QAI Hub models

Technical Notes:
- QAI Hub models contain preprocessing (e.g., ImageNet normalization) with
  mean/std tensors that may stay on CPU even when model is on CUDA
- This module applies a device patch to handle CPU/GPU tensor transfers in preprocessing
"""

import torch
from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.utils.evaluate import get_deterministic_sample
from ONNXRegression.features.torch.utils import ensure_device_patch


def eval_pytorch_model(
    model: torch.nn.Module,
    qai_hub_model,
    dataset_name: str,
    num_samples: int = 200,
    batch_size: int = 32,
) -> float:
    """
    Evaluate PyTorch model accuracy on a dataset.

    This function provides a unified interface for accuracy evaluation on PyTorch models,
    using the qai_hub_models dataset infrastructure.

    Args:
        model: PyTorch model to evaluate (torch.nn.Module or QAI Hub model)
        qai_hub_model: QAI Hub model instance (provides pre/post-processing reference)
        dataset_name: Name of the dataset (e.g., "imagenet", "coco")
        num_samples: Number of samples to evaluate (default: 200)

    Returns:
        Top-1 accuracy as percentage in range [0, 100]
    """
    ensure_device_patch()
    model.eval()

    dataset = get_dataset_from_name(dataset_name, DatasetSplit.VAL)
    dataloader = get_deterministic_sample(
        dataset, num_samples=num_samples, samples_per_job=batch_size
    )

    correct = 0
    total = 0

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.no_grad():
        for sample in dataloader:
            if isinstance(sample, (list, tuple)) and len(sample) == 2:
                inputs, labels = sample
            else:
                inputs = sample
                labels = None

            # Move inputs to model device
            if isinstance(inputs, dict):
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                outputs = model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                inputs = tuple(
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs
                )
                outputs = model(*inputs)
            else:
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                outputs = model(inputs)

            if isinstance(outputs, dict):
                outputs = outputs.get(
                    "logits", outputs.get("output", list(outputs.values())[0])
                )

            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

    if total == 0:
        return 0.0

    accuracy_pct = (correct / total) * 100.0
    return round(accuracy_pct, 3)
