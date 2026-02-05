# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-ir related utility functions"""

import onnx_ir
import numpy as np


def get_constant_singleton_value(value: onnx_ir.Value | None) -> float | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    numpy_value = get_constant_as_array(value)

    if numpy_value is None or numpy_value.size != 1:
        return None

    return float(numpy_value.flatten()[0])


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
