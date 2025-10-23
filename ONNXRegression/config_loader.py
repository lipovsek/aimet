# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Configuration Loader for AIMET ONNX Regression

This module provides simple utilities for loading and merging YAML configuration
files in the new hierarchical config system.

Key Features:
- Load and merge configs in precedence order: defaults → profile → model → test
- Discover available tests in model configs
- List all available models
- Simple dict.update() merging (no deep merge complexity)
- Clear error messages showing available options


Usage:
    from ONNXRegression.config_loader import load_config, list_tests

    # Load a specific test configuration
    config = load_config("models/resnet50.yaml", "quantsim_int8", "nightly")

    # Discover available tests
    tests = list_tests("models/resnet50.yaml")

"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional


# ==================== Configuration Loading ====================


def load_config(
    model_yaml: str, test_name: str, profile: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load and merge configuration from multiple sources.

    This is the main entry point for config loading. It merges configs in order
    of increasing precedence:
        1. _defaults.yaml (base settings for all tests)
        2. _profiles/{profile}.yaml (runtime scenario settings)
        3. models/{model}.yaml (model-level settings)
        4. Test config within model file (test-specific settings)

    Later configs override earlier ones using simple dict.update().

    Args:
        model_yaml: Model config path relative to configs/ directory.
                   Example: "models/resnet50.yaml"
        test_name: Name of test to load from model's tests list.
                  Example: "quantsim_int8"
        profile: Optional profile name (e.g., "nightly", "smoke").
                If None, only defaults + model + test are merged.

    Returns:
        Merged configuration dictionary ready for test execution.
        Contains all settings needed by runner.py and feature runners.

    Raises:
        FileNotFoundError: If any config file doesn't exist
        ValueError: If test_name not found in model config

    Examples:
        >>> # Load with nightly profile (reduced samples, no QNN)
        >>> config = load_config("models/resnet50.yaml", "quantsim_int8", "nightly")
        >>> print(config['calib_samples'])  # 128 (from nightly profile)
        >>> print(config['qnn_options'])    # None (disabled by nightly)

        >>> # Load without profile (use defaults)
        >>> config = load_config("models/resnet50.yaml", "quantsim_int8")
        >>> print(config['calib_samples'])  # 256 (from defaults)
    """
    base_dir = Path("ONNXRegression/configs")

    # Step 1: Load defaults (foundation for all configs)
    defaults_path = base_dir / "_defaults.yaml"
    if not defaults_path.exists():
        raise FileNotFoundError(
            f"Defaults file not found: {defaults_path}\n"
            f"This file is required and should contain base settings."
        )

    with open(defaults_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Step 2: Merge profile (if specified)
    if profile:
        profile_path = base_dir / "_profiles" / f"{profile}.yaml"

        if not profile_path.exists():
            # Show available profiles for helpful error message
            available_profiles = [
                p.stem for p in (base_dir / "_profiles").glob("*.yaml")
            ]
            raise FileNotFoundError(
                f"Profile '{profile}' not found: {profile_path}\n"
                f"Available profiles: {', '.join(available_profiles) or 'none'}"
            )

        with open(profile_path, "r", encoding="utf-8") as f:
            prof_config = yaml.safe_load(f) or {}

        # Remove 'extends' keyword if present (we don't process it)
        prof_config.pop("extends", None)

        # Merge profile on top of defaults (simple dict update)
        config.update(prof_config)

    # Step 3: Load model configuration
    model_path = base_dir / model_yaml

    if not model_path.exists():
        # Show available models for helpful error message
        available_models = [
            f"models/{m.name}" for m in (base_dir / "models").glob("*.yaml")
        ]
        raise FileNotFoundError(
            f"Model config not found: {model_path}\n"
            f"Available models: {', '.join(available_models) or 'none'}"
        )

    with open(model_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f) or {}

    # Remove 'extends' keyword if present
    model_config.pop("extends", None)

    # Extract tests list (required)
    tests = model_config.pop("tests", [])

    if not tests:
        raise ValueError(
            f"Model config has no 'tests' list: {model_path}\n"
            f"Each model must define at least one test."
        )

    # Merge model-level settings (everything except 'tests')
    config.update(model_config)

    # Step 4: Find and merge specific test configuration
    test_config = next((t for t in tests if t.get("name") == test_name), None)

    if test_config is None:
        # Show available tests for helpful error message
        available_tests = [t.get("name", "unnamed") for t in tests]
        raise ValueError(
            f"Test '{test_name}' not found in {model_yaml}\n"
            f"Available tests: {', '.join(available_tests)}"
        )

    # Merge test-specific settings
    config.update(test_config)

    return config


# ==================== Test Discovery ====================


def list_tests(model_yaml: str) -> List[str]:
    """
    List all test names defined in a model configuration.

    This is useful for:
    - Discovering what tests exist in a model
    - Filtering tests in suite_runner.py
    - Generating error messages with available options

    Args:
        model_yaml: Model config path relative to configs/ directory.
                   Example: "models/resnet50.yaml"

    Returns:
        List of test names (e.g., ['quantsim_int8', 'lite_mp_25'])

    Raises:
        FileNotFoundError: If model config doesn't exist

    Example:
        >>> tests = list_tests("models/resnet50.yaml")
        >>> print(tests)
        ['quantsim_int8', 'quantsim_int8_int16', 'lite_mp_25', 'adaround_500']
    """
    base_dir = Path("ONNXRegression/configs")
    model_path = base_dir / model_yaml

    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_path}")

    with open(model_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f) or {}

    tests = model_config.get("tests", [])
    return [t.get("name", "unnamed") for t in tests]


def list_models() -> List[str]:
    """
    List all available model configurations.

    Scans the configs/models/ directory and returns paths to all model YAMLs.

    Returns:
        List of model config paths (e.g., ['models/resnet50.yaml', ...])

    Example:
        >>> models = list_models()
        >>> for model in models:
        ...     tests = list_tests(model)
        ...     print(f"{model}: {len(tests)} tests")
        models/resnet50.yaml: 4 tests
        models/mobilenetv2.yaml: 2 tests
    """
    base_dir = Path("ONNXRegression/configs/models")

    if not base_dir.exists():
        return []

    return sorted([f"models/{f.name}" for f in base_dir.glob("*.yaml")])


# ==================== Validation ====================


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that a configuration has required fields.

    Performs basic validation to catch common errors early.
    More detailed validation happens in feature runners.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required fields are missing or invalid

    Example:
        >>> config = load_config("models/resnet50.yaml", "quantsim_int8")
        >>> validate_config(config)  # Raises if invalid
    """
    # Required fields that must be present
    required_fields = ["model_name", "feature", "framework"]

    missing = [field for field in required_fields if field not in config]

    if missing:
        raise ValueError(
            f"Configuration missing required fields: {', '.join(missing)}\n"
            f"These fields must be defined in either defaults, profile, model, or test config."
        )

    # Validate feature is supported
    supported_features = ["quantsim", "lite_mp", "adaround"]
    if config["feature"] not in supported_features:
        raise ValueError(
            f"Unsupported feature: {config['feature']}\n"
            f"Supported features: {', '.join(supported_features)}"
        )

    # Validate framework
    if config["framework"] != "onnx":
        raise ValueError(
            f"Unsupported framework: {config['framework']}\n"
            f"This tool only supports ONNX"
        )
