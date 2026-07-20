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

import itertools
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qai_hub_models.datasets import BaseDataset, DatasetSplit, instantiate_dataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.helpers import (
    get_deterministic_sample,
    DatasetFromIOTuples,
)

from AIMETRegression.features.torch.utils import ensure_device_patch


def load_torch_dataset(qai_hub_model: BaseModel, dataset_cls: type[BaseDataset]):
    """
    Load a dataset for torch evaluation. Call once and pass the result to
    eval_pytorch_model(dataset=...) to avoid reloading on every call.
    """
    input_spec = qai_hub_model.get_input_spec()
    return instantiate_dataset(dataset_cls, DatasetSplit.VAL, input_spec)


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
    dataset_cls: type[BaseDataset],
    num_samples: int = 200,
    batch_size: int = 32,
    dataset=None,
) -> float:
    """
    Evaluate PyTorch model accuracy on a dataset.

    This function isolates the local torch evaluation path from qai_hub_models'
    evaluate_on_dataset(), adding GPU device handling. It uses the same evaluator
    infrastructure to ensure consistent accuracy computation.

    Args:
        model: PyTorch model to evaluate (torch.nn.Module, can be on GPU)
        qai_hub_model: QAI Hub model instance (BaseModel, provides evaluator and input_spec)
        dataset_cls: Dataset class to evaluate on (e.g., ImagenetDataset)
        num_samples: Number of samples to evaluate (default: 200)
        batch_size: Batch size for inference (default: 32). Higher values
            amortize CPU↔GPU transfer overhead but use more GPU memory.
        dataset: Pre-loaded dataset from load_torch_dataset(). If provided,
            skips dataset loading — avoids redundant disk I/O when this
            function is called multiple times (calibration, eval, metrics).

    Returns:
        Top-1 accuracy as float in range [0, 1]

    Example:
        >>> accuracy = eval_pytorch_model(
        ...     torch_model,
        ...     resnet50_qai_model,
        ...     ImagenetDataset,
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
    if dataset is not None:
        source_torch_dataset = dataset
    else:
        input_spec = qai_hub_model.get_input_spec()
        source_torch_dataset = instantiate_dataset(
            dataset_cls, DatasetSplit.VAL, input_spec
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

    print(f"Evaluating on {num_samples} samples (batch_size={batch_size}).")

    # Get evaluator from qai_hub_model
    evaluator = qai_hub_model.get_evaluator()

    # Ensure model is in eval mode
    try:
        model.eval()
    except NotImplementedError:
        # torch.fx.GraphModules generated from ExportedProgram
        # doesn't support .eval() method yet
        pass

    # --- Addition: Determine model device for GPU handling ---
    try:
        device = next(itertools.chain(model.parameters(), model.buffers())).device
    except StopIteration:
        device = torch.device("cpu")

    # --- From evaluate(): Local model evaluation loop ---
    model_batch_size = batch_size

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

                # Unwrap single-element tuples (e.g., HuggingFace models
                # with return_dict=False return (logits,) instead of logits)
                if isinstance(batch_output, tuple) and len(batch_output) == 1:
                    batch_output = batch_output[0]

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
