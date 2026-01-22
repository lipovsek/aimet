# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
QAI Hub Models Loader

This module provides utilities for loading models from the Qualcomm AI Hub Models
repository along with their associated datasets and input specifications.

Key Functionality:
- Dynamic model loading from qai_hub_models package
- Automatic dataset resolution from model metadata
- Input specification extraction for pipeline compatibility
- Graceful handling of various model attribute formats
- CI/CD compatibility (auto-accepts git clones for external repos)

Design Philosophy:
QAI Hub Models have varying structures and metadata formats. This module
provides a unified interface that handles these variations, ensuring
consistent access to models regardless of their specific implementation.
"""

import os
import importlib
from typing import Tuple, Any, Optional, Union, List

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.utils.evaluate import get_deterministic_sample
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.testing import always_answer_prompts


# ==================== Utility Functions ====================


def _call_if_callable(x: Any) -> Any:
    """
    Call object if it's callable, otherwise return as-is.

    Many QAI Hub model attributes are properties or methods that need
    to be invoked to get their actual values. This utility handles both
    cases uniformly.

    Args:
        x: Object that might be callable (property, method, or value)

    Returns:
        Result of calling x() if callable, otherwise x itself

    Example:
        >>> class Model:
        ...     @property
        ...     def dataset(self): return "imagenet"
        >>> m = Model()
        >>> _call_if_callable(m.dataset)  # Returns "imagenet"
        >>> _call_if_callable("coco")     # Returns "coco"
    """
    return x() if callable(x) else x


def _to_name_str(x: Any) -> str:
    """
    Convert various value types to a string name.

    QAI Hub Models use different representations for names:
    - Plain strings: returned as-is
    - Enums with .value attribute: extract the value
    - Enums with .name attribute: extract the name (then .value if nested)
    - Other objects: convert to string

    Args:
        x: Value to convert (string, enum, or other object)

    Returns:
        String representation of the name

    Technical Note:
        This handles the variety of enum types used in qai_hub_models,
        including both Python enums and custom enum-like classes.
    """
    # Already a string
    if isinstance(x, str):
        return x

    # Has .value attribute (common for enums)
    if hasattr(x, "value"):
        return x.value

    # Has .name attribute (alternative enum format)
    if hasattr(x, "name"):
        name = x.name
        # Handle nested enum (name itself might have .value)
        return name.value if hasattr(name, "value") else str(name)

    # Fallback: convert to string
    return str(x)


def _pick_model_cls(module) -> type:
    """
    Find and return the appropriate model class from a module.

    QAI Hub Models modules may contain the model class under various names.
    This function searches for the correct class using multiple strategies:
    1. Look for common names (Model, BaseModel)
    2. Search all module attributes for BaseModel subclasses

    Args:
        module: Python module containing the model class

    Returns:
        Model class (subclass of BaseModel)

    Raises:
        RuntimeError: If no suitable model class is found

    Search Strategy:
        1. Check for 'Model' attribute
        2. Check for 'BaseModel' attribute (if not the base class itself)
        3. Scan all module attributes for BaseModel subclasses
    """
    # Strategy 1: Try common explicit names
    for class_name in ("Model", "BaseModel"):
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            # Ensure it's a BaseModel subclass (not BaseModel itself)
            if (
                isinstance(cls, type)
                and issubclass(cls, BaseModel)
                and cls is not BaseModel
            ):
                return cls

    # Strategy 2: Search all attributes for BaseModel subclasses
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        try:
            # Check if it's a class and subclass of BaseModel
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
            ):
                return obj
        except TypeError:
            # issubclass raises TypeError if obj is not a class
            pass

    # No suitable class found
    raise RuntimeError(
        f"No BaseModel subclass found in module {module.__name__}. "
        f"Ensure the model module exports a class inheriting from BaseModel."
    )


def _resolve_dataset_name(model: BaseModel) -> str:
    """
    Intelligently resolve dataset name from model metadata.

    QAI Hub Models store dataset information in various locations and formats:
    - calibration_dataset_name: Primary dataset for quantization calibration
    - eval_datasets: List of datasets for evaluation (we use the first)

    This function tries multiple approaches to find a valid dataset name,
    handling properties, methods, enums, and nested structures.

    Args:
        model: QAI Hub model instance

    Returns:
        Dataset name as string (e.g., "imagenet", "coco", "cifar10")

    Raises:
        RuntimeError: If dataset name cannot be resolved

    Resolution Order:
        1. Try calibration_dataset_name (most specific)
        2. Try first entry in eval_datasets list
        3. Raise error if neither found
    """
    # First attempt: calibration_dataset_name
    # This is the preferred source as it's most specific to our use case
    calibration_name = _call_if_callable(
        getattr(model, "calibration_dataset_name", None)
    )
    if calibration_name:
        return _to_name_str(calibration_name)

    # Second attempt: eval_datasets
    # Some models only provide evaluation datasets, not calibration-specific ones
    eval_datasets = _call_if_callable(getattr(model, "eval_datasets", None))

    if isinstance(eval_datasets, (list, tuple)) and eval_datasets:
        # Take the first dataset from the list
        first_dataset = eval_datasets[0]

        # Handle nested structure (dataset object with name attribute)
        if hasattr(first_dataset, "name"):
            return _to_name_str(first_dataset.name)
        else:
            return _to_name_str(first_dataset)

    # Unable to resolve dataset name
    raise RuntimeError(
        f"Unable to resolve dataset name from model {type(model).__name__}. "
        f"Model must provide either 'calibration_dataset_name' or 'eval_datasets' attribute."
    )


# ==================== Main API ====================


def load_model_data(model_name: str) -> Tuple[BaseModel, Any, dict, Any]:
    """
    Load a model from QAI Hub Models with its associated dataset and specifications.

    This is the main entry point for loading models in the AIMET regression pipeline.
    It handles the complete setup process:
    1. Dynamically import the model module
    2. Find and instantiate the model class
    3. Load pretrained weights
    4. Set up the dataset and dataloader
    5. Extract input specifications

    Args:
        model_name: Name of the model in qai_hub_models
                   (e.g., "resnet50", "mobilenet_v2", "yolov5s")

    Returns:
        Tuple containing:
            - model: Instantiated and pretrained model object
            - dataset: Dataset object for evaluation
            - input_spec: Input specification dictionary
            - dataloader: Dataloader with deterministic sampling

    Raises:
        ModuleNotFoundError: If model module doesn't exist
        RuntimeError: If model class cannot be found or dataset cannot be resolved

    Example:
        >>> model, dataset, input_spec, dataloader = load_model_data("resnet50")
        >>> print(f"Model loaded: {type(model).__name__}")
        >>> print(f"Dataset: {dataset.name}")
        >>> print(f"Input shape: {input_spec['image'].shape}")

    Technical Notes:
        - Models are loaded with pretrained weights via from_pretrained()
        - Dataset split is always VALIDATION for evaluation consistency
        - Dataloader uses deterministic sampling for reproducibility
        - Default to 100 samples for initial testing (can be overridden)
        - Git clone prompts are automatically accepted in CI/CD environments
    """
    # Wrap the entire model loading process to auto-accept git clone prompts
    # This is critical for CI/CD where interactive prompts would block execution
    with always_answer_prompts(True):
        # ============ Step 1: Import Model Module ============
        try:
            module = importlib.import_module(f"qai_hub_models.models.{model_name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Model '{model_name}' not found in qai_hub_models. "
                f"Ensure the model name is correct and qai_hub_models is installed. "
                f"Available models can be found at: https://github.com/quic/ai-hub-models"
            ) from e

        # ============ Step 2: Find Model Class ============
        model_cls = _pick_model_cls(module)

        # ============ Step 3: Instantiate Model ============
        # Load with pretrained weights (standard for evaluation)
        model: BaseModel = model_cls.from_pretrained()

        # ============ Step 4: Resolve Dataset ============
        dataset_name = _resolve_dataset_name(model)

        # Load the validation split of the dataset
        # We use validation split for all evaluation to avoid training data leakage
        dataset = get_dataset_from_name(
            dataset_name,
            DatasetSplit.VAL,  # Always use validation split for evaluation
        )

        # ============ Step 5: Create Dataloader ============
        # Deterministic sampling ensures reproducible results
        dataloader = get_deterministic_sample(
            dataset,
            num_samples=100,  # Default sample size for initial testing
            samples_per_job=100,  # Process all samples in one batch
        )

        # ============ Step 6: Get Input Specification ============
        # Input spec defines the expected input format (shape, dtype, etc.)
        input_spec = model.get_input_spec()

    return model, dataset, input_spec, dataloader
