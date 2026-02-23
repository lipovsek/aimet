# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-ir related utility functions"""

import onnx_ir
import numpy as np


def get_constant_singleton_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> float | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    numpy_value = get_constant_or_attribute_value(value)

    if numpy_value is None or numpy_value.size != 1:
        return None

    return numpy_value.flatten()[0].item()


def get_constant_as_array(value: onnx_ir.Value | None) -> np.ndarray | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    if value is None:
        return None

    const_value = onnx_ir.convenience.get_const_tensor(value)
    if const_value is None:
        return None

    return const_value.numpy()


def get_constant_or_attribute_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> None | np.ndarray:
    """Get the constant value from an ONNX IR Value or Attr, if it exists."""
    if value is None:
        return None
    if isinstance(value, onnx_ir.Value):
        return get_constant_as_array(value)
    if isinstance(value, onnx_ir.Attr):
        return np.asarray(value.value)
    raise RuntimeError(f"Received unexpected type for value: {type(value)}")
