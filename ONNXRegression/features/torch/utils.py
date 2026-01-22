# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Device Patch Utility for QAI Hub Models

QAI Hub model wrappers contain preprocessing (e.g., ImageNet normalization) with
mean/std tensors that may remain on CPU even when the model is moved to CUDA.
This causes device mismatch errors during AIMET calibration.

This module provides a transparent patch that wraps preprocessing functions to
handle device transfers automatically. The patch is applied globally to ALL
modules that have imported the target functions.

Usage:
    from ONNXRegression.features.torch.utils import ensure_device_patch

    # Call early in your code, after model imports
    ensure_device_patch()
"""

from __future__ import annotations

import sys
import functools
from typing import Callable, Dict, List

import torch

_PATCH_APPLIED = False
_ORIGINAL_FUNCTIONS: Dict[str, Callable] = {}
_PATCHED_MODULES: List[str] = []


def _make_device_aware_wrapper(original_fn: Callable) -> Callable:
    """
    Create a wrapper that handles CPU/GPU device transfers transparently.

    Strategy:
    - If input tensor is on CUDA, move to CPU for processing
    - Call original function on CPU
    - Move result back to original CUDA device

    This avoids device mismatch errors when preprocessing tensors have
    hardcoded CPU mean/std values.
    """

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        # Find tensor arguments and their device
        tensor_args = []
        input_device = None

        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensor_args.append((i, arg))
                if input_device is None:
                    input_device = arg.device

        # If on CUDA, move all tensor args to CPU
        if input_device is not None and input_device.type == "cuda":
            new_args = list(args)
            for idx, tensor in tensor_args:
                new_args[idx] = tensor.cpu()
            args = tuple(new_args)

            # Also handle tensor kwargs
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    new_kwargs[k] = v.cpu()
                else:
                    new_kwargs[k] = v
            kwargs = new_kwargs

        # Call original function
        result = original_fn(*args, **kwargs)

        # Move result back to original device if needed
        if input_device is not None and input_device.type == "cuda":
            if isinstance(result, torch.Tensor):
                result = result.to(input_device)

        return result

    return wrapper


def apply_device_patch() -> None:
    """
    Apply device-aware patches to QAI Hub preprocessing functions.

    This patches normalize_image_torchvision in:
    1. The source module (qai_hub_models.utils.image_processing)
    2. ALL other modules that have already imported this function

    This comprehensive approach ensures the patch works regardless of
    import order or which module calls the function.
    """
    global _PATCH_APPLIED, _ORIGINAL_FUNCTIONS, _PATCHED_MODULES

    if _PATCH_APPLIED:
        return

    try:
        from qai_hub_models.utils import image_processing

        # Check if already patched by another mechanism
        if hasattr(image_processing, "_device_patched"):
            _PATCH_APPLIED = True
            return

        if not hasattr(image_processing, "normalize_image_torchvision"):
            return

        # Store original function
        original_fn = image_processing.normalize_image_torchvision
        _ORIGINAL_FUNCTIONS["normalize_image_torchvision"] = original_fn

        # Create patched version
        patched_fn = _make_device_aware_wrapper(original_fn)

        # Patch the source module
        image_processing.normalize_image_torchvision = patched_fn
        image_processing._device_patched = True
        _PATCHED_MODULES.append("qai_hub_models.utils.image_processing")

        # Find and patch ALL modules that have imported this function
        # This is critical because some model modules do:
        #   from qai_hub_models.utils.image_processing import normalize_image_torchvision
        # which creates a separate reference that won't be updated by patching the source
        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
            try:
                if hasattr(module, "normalize_image_torchvision"):
                    current_fn = getattr(module, "normalize_image_torchvision")
                    # Only patch if it's the original function (not already patched)
                    if current_fn is original_fn:
                        setattr(module, "normalize_image_torchvision", patched_fn)
                        _PATCHED_MODULES.append(module_name)
            except Exception:
                # Some modules may raise on attribute access
                pass

        _PATCH_APPLIED = True
        print(
            f"[DevicePatch] Patched normalize_image_torchvision in {len(_PATCHED_MODULES)} modules"
        )

    except ImportError:
        # qai_hub_models not installed
        pass


def remove_device_patch() -> None:
    """
    Remove the device patch and restore original functions.

    Useful for testing or when you need to restore original behavior.
    """
    global _PATCH_APPLIED, _ORIGINAL_FUNCTIONS, _PATCHED_MODULES

    if not _PATCH_APPLIED:
        return

    try:
        from qai_hub_models.utils import image_processing

        # Restore original function to all patched modules
        if "normalize_image_torchvision" in _ORIGINAL_FUNCTIONS:
            original_fn = _ORIGINAL_FUNCTIONS["normalize_image_torchvision"]

            for module_name in _PATCHED_MODULES:
                try:
                    module = sys.modules.get(module_name)
                    if module and hasattr(module, "normalize_image_torchvision"):
                        setattr(module, "normalize_image_torchvision", original_fn)
                except Exception:
                    pass

            # Remove the patched flag
            if hasattr(image_processing, "_device_patched"):
                delattr(image_processing, "_device_patched")

        _ORIGINAL_FUNCTIONS.clear()
        _PATCHED_MODULES.clear()
        _PATCH_APPLIED = False

    except ImportError:
        pass


def ensure_device_patch() -> None:
    """
    Ensure device patch is applied. Safe to call multiple times.

    This is the recommended entry point - call this early in your code
    after importing the model to ensure all modules are patched.
    """
    if not _PATCH_APPLIED:
        apply_device_patch()


def is_patch_applied() -> bool:
    """Check if the device patch is currently applied."""
    return _PATCH_APPLIED
