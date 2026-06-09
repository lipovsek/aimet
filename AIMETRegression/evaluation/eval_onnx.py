# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
ONNX Model Evaluation Module

This module provides utilities for evaluating ONNX models using ONNXRuntime.
It handles accuracy computation for various QAI Hub models.

Key functionality:
- Standardized accuracy evaluation interface
- Support for both file paths and existing ORT sessions

"""

import onnxruntime as ort
from qai_hub_models.datasets import BaseDataset
from qai_hub_models.utils.evaluate import evaluate_session_on_dataset


def eval_onnx_model(
    session_or_path, model, dataset_cls: type[BaseDataset], num_samples: int = 200
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
        dataset_cls: Dataset class to evaluate on (e.g., ImagenetDataset)
        num_samples: Number of samples to evaluate (default: 200)

    Returns:
        Top-1 accuracy as float in range [0, 1]

    Example:
        >>> accuracy = eval_onnx_model(
        ...     "model.onnx",
        ...     resnet50_model,
        ...     ImagenetDataset,
        ...     num_samples=1000
        ... )
        >>> print(f"Accuracy: {accuracy:.2%}")
        Accuracy: 75.60%
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Determine execution providers - prefer CUDA if available
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Create session if path was provided
    if isinstance(session_or_path, ort.InferenceSession):
        session = session_or_path
    else:
        # Load ONNX model from file
        session = ort.InferenceSession(
            str(session_or_path), sess_options=sess_options, providers=providers
        )

    # Evaluate using QAI Hub's standardized evaluation
    accuracy, _ = evaluate_session_on_dataset(
        session, model, dataset_cls, num_samples=num_samples
    )

    # Convert to float (sometimes returns numpy scalar)
    return float(accuracy)
