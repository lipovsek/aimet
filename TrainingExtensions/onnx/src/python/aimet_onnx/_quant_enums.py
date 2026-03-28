# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared quantization enums used by Cython modules.

This module provides IntEnum classes that are shared between _libpymo and
libquant_info Cython modules. Using IntEnum allows direct integer conversion
which is needed when passing values to/from C++ code.
"""

from enum import IntEnum


class TensorQuantizerOpMode(IntEnum):
    """Operation mode for quantization."""

    updateStats = 0
    oneShotQuantizeDequantize = 1
    quantizeDequantize = 2
    passThrough = 3


class QuantizationMode(IntEnum):
    """Quantization scheme."""

    QUANTIZATION_TF = 0
    QUANTIZATION_TF_ENHANCED = 1
    QUANTIZATION_RANGE_LEARNING = 2
    QUANTIZATION_PERCENTILE = 3
    QUANTIZATION_MSE = 4
    QUANTIZATION_ENTROPY = 5


class ComputationMode(IntEnum):
    """Computation mode (CPU or GPU)."""

    COMP_MODE_CPU = 0
    COMP_MODE_GPU = 1


class LayerInOut(IntEnum):
    """Layer input/output indicator."""

    LAYER_INPUT = 0
    LAYER_OUTPUT = 1


class RoundingMode(IntEnum):
    """Rounding mode for quantization."""

    ROUND_NEAREST = 0
    ROUND_STOCHASTIC = 1
