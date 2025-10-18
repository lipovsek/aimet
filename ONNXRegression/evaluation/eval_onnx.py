# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
ONNX Model Evaluation Module

This module provides utilities for evaluating ONNX models using ONNXRuntime.
It handles dataset resolution and accuracy computation for various QAI Hub models.

Key functionality:
- Automatic dataset name resolution from model metadata
- Standardized accuracy evaluation interface
- Support for both file paths and existing ORT sessions

"""

import onnxruntime as ort
from qai_hub_models.utils.evaluate import evaluate_session_on_dataset


def _extract_value(x):
    """
    Extract the underlying value from various wrapper types.

    Some QAI Hub models use enum or wrapper objects for metadata.
    This function extracts the actual value.

    Args:
        x: Value that might be wrapped (enum, object with .value, etc.)

    Returns:
        Underlying value (usually a string)
    """
    return x.value if hasattr(x, "value") else x


def _call_if_callable(x):
    """
    Call object if it's callable, otherwise return as-is.

    Some model attributes are properties or methods that need to be called.

    Args:
        x: Object that might be callable

    Returns:
        Result of calling x if callable, otherwise x itself
    """
    return x() if callable(x) else x


def resolve_dataset_name(model) -> str:
    """
    Intelligently resolve dataset name from model metadata.

    QAI Hub models store dataset information in various ways:
    - calibration_dataset_name: Primary dataset for calibration
    - eval_datasets: List of evaluation datasets

    This function tries multiple approaches to find a valid dataset name.

    Args:
        model: QAI Hub model instance

    Returns:
        Dataset name as string (e.g., "imagenet", "coco")

    Raises:
        RuntimeError: If dataset name cannot be resolved
    """
    dataset_name = None

    # Try calibration dataset first (most specific)
    if hasattr(model, "calibration_dataset_name"):
        value = _call_if_callable(model.calibration_dataset_name)
        dataset_name = _extract_value(value)

    # Fall back to evaluation datasets
    if not dataset_name and hasattr(model, "eval_datasets"):
        eval_datasets = _call_if_callable(model.eval_datasets)
        if eval_datasets:
            # Take the first dataset from the list
            first_dataset = eval_datasets[0]
            # Handle nested attributes
            if hasattr(first_dataset, "name"):
                dataset_name = _extract_value(first_dataset.name)
            else:
                dataset_name = _extract_value(first_dataset)

    # Validate we got a string
    if not isinstance(dataset_name, str):
        raise RuntimeError(
            f"Could not resolve dataset name to string. Got: {type(dataset_name)}"
        )

    return dataset_name


def eval_onnx_model(
    session_or_path, model, dataset_name: str, num_samples: int = 200
) -> float:
    """
    Evaluate ONNX model accuracy on a dataset.

    This function provides a unified interface for accuracy evaluation,
    accepting either an ONNX file path or an existing ORT session.

    Args:
        session_or_path: Either:
            - Path to ONNX model file (str or Path)
            - Existing ort.InferenceSession instance
        model: QAI Hub model instance (provides pre/post-processing)
        dataset_name: Name of the dataset (e.g., "imagenet")
        num_samples: Number of samples to evaluate (default: 200)

    Returns:
        Top-1 accuracy as float in range [0, 1]

    Example:
        >>> accuracy = eval_onnx_model(
        ...     "model.onnx",
        ...     resnet50_model,
        ...     "imagenet",
        ...     num_samples=1000
        ... )
        >>> print(f"Accuracy: {accuracy:.2%}")
        Accuracy: 75.60%
    """
    # Create session if path was provided
    if isinstance(session_or_path, ort.InferenceSession):
        session = session_or_path
    else:
        # Load ONNX model from file
        session = ort.InferenceSession(str(session_or_path))

    # Evaluate using QAI Hub's standardized evaluation
    accuracy, _ = evaluate_session_on_dataset(
        session, model, dataset_name, num_samples=num_samples
    )

    # Convert to float (sometimes returns numpy scalar)
    return float(accuracy)
