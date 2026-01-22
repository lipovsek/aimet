# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
PyTorch Model Evaluation Module

Provides utilities for evaluating PyTorch models using qai_hub_models evaluation
infrastructure. This isolates the torch evaluation path from qai_hub_models'
evaluate() function, adding GPU device handling.

Key functionality:
- Native PyTorch model evaluation using qai_hub_models standardized evaluation
- Uses the same evaluator infrastructure as evaluate_on_dataset
- Consistent interface and return values with eval_onnx.py
- GPU-aware: handles CPU↔GPU tensor transfers automatically
- Supports multiple inputs and multiple outputs

Notes:
- Isolated from qai_hub_models.utils.evaluate.evaluate() local model path
- Uses DatasetFromIOTuples for rebatching to model_batch_size=1
- Gets evaluator from qai_hub_model.get_evaluator() for proper metric computation
- Applies device patch to handle CPU/GPU tensor transfers in preprocessing
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate import (
    get_deterministic_sample,
    DatasetFromIOTuples,
)

from ONNXRegression.features.torch.utils import ensure_device_patch


def _torch_io_to_tuple(val) -> tuple[torch.Tensor, ...]:
    """
    Convert torch model I/O of any type to a tuple of tensors.

    Copied from qai_hub_models.utils.evaluate.evaluate() inner function.
    """
    if isinstance(val, tuple):
        return val
    if isinstance(val, list):
        return tuple(val)
    return (val,)


def eval_pytorch_model(
    model: torch.nn.Module,
    qai_hub_model: BaseModel,
    dataset_name: str,
    num_samples: int = 200,
) -> float:
    """
    Evaluate PyTorch model accuracy on a dataset.

    This function isolates the local torch evaluation path from qai_hub_models'
    evaluate_on_dataset(), adding GPU device handling. It uses the same evaluator
    infrastructure to ensure consistent accuracy computation.

    Args:
        model: PyTorch model to evaluate (torch.nn.Module, can be on GPU)
        qai_hub_model: QAI Hub model instance (BaseModel, provides evaluator and input_spec)
        dataset_name: Name of the dataset (e.g., "imagenet", "coco")
        num_samples: Number of samples to evaluate (default: 200)

    Returns:
        Top-1 accuracy as float in range [0, 1]

    Example:
        >>> accuracy = eval_pytorch_model(
        ...     torch_model,
        ...     resnet50_qai_model,
        ...     "imagenet",
        ...     num_samples=1000
        ... )
        >>> print(f"Accuracy: {accuracy:.2%}")
        Accuracy: 75.60%

    Note:
        This function applies a device patch to handle QAI Hub models that have
        preprocessing tensors (mean/std) on CPU while the model is on CUDA.
    """
    # Apply device patch for preprocessing compatibility
    ensure_device_patch()

    # Validate qai_hub_model is a BaseModel
    assert isinstance(qai_hub_model, BaseModel), (
        "Evaluation requires a BaseModel instance for get_input_spec() and get_evaluator()."
    )

    # --- From evaluate_on_dataset(): Setup dataloader ---
    input_spec = qai_hub_model.get_input_spec()
    source_torch_dataset = get_dataset_from_name(
        dataset_name, DatasetSplit.VAL, input_spec
    )

    # Validate inputs
    if num_samples < 1 and num_samples != -1:
        raise ValueError("num_samples must be positive or -1.")
    if num_samples > len(source_torch_dataset):
        raise ValueError(
            f"Requested {num_samples} samples when dataset only has {len(source_torch_dataset)}."
        )

    # Create deterministic dataloader (samples_per_job=None means all in one batch)
    dataloader = get_deterministic_sample(source_torch_dataset, num_samples, None)

    print(f"Evaluating on {num_samples} samples.")

    # Get evaluator from qai_hub_model
    evaluator = qai_hub_model.get_evaluator()

    # Ensure model is in eval mode
    model.eval()

    # --- Addition: Determine model device for GPU handling ---
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # --- From evaluate(): Local model evaluation loop ---
    model_batch_size = 1  # Use batch_size=1 for inference (matches compiled models)

    with torch.no_grad():
        for sample in dataloader:
            model_inputs, ground_truth_values, *_ = sample

            # From evaluate(): Convert to tuple format
            model_inputs = _torch_io_to_tuple(model_inputs)
            ground_truth_values = _torch_io_to_tuple(ground_truth_values)

            # From evaluate(): Rebatch to model_batch_size using DatasetFromIOTuples
            local_dataset = DatasetFromIOTuples(model_inputs, ground_truth_values)
            local_dataloader = DataLoader(local_dataset, model_batch_size)

            for local_sample in tqdm(local_dataloader, disable=True):
                local_model_inputs, local_ground_truth_values, *_ = local_sample

                # --- Addition: Move inputs to model's device (GPU) ---
                if isinstance(local_model_inputs, (list, tuple)):
                    local_model_inputs = tuple(x.to(device) for x in local_model_inputs)
                    batch_output = model(*local_model_inputs)
                else:
                    local_model_inputs = local_model_inputs.to(device)
                    batch_output = model(local_model_inputs)

                # --- Addition: Move outputs to CPU for evaluator ---
                if isinstance(batch_output, torch.Tensor):
                    batch_output = batch_output.cpu()
                elif isinstance(batch_output, tuple):
                    batch_output = tuple(
                        o.cpu() if isinstance(o, torch.Tensor) else o
                        for o in batch_output
                    )

                # From evaluate(): Add batch to evaluator
                evaluator.add_batch(batch_output, local_ground_truth_values)

    # Get accuracy score
    accuracy = evaluator.get_accuracy_score()

    # Convert to float (sometimes returns numpy scalar)
    return float(accuracy)
