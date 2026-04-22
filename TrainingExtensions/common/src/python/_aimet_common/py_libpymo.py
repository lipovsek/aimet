# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python3

"""Creating an alias for function/classes/methods to use AIMET without MO library"""

import enum
from enum import IntEnum

IMPORT_ERROR: ImportError = ImportError()


def _error_message():
    return (
        f"libpymo import failed with the following error:\n\n{IMPORT_ERROR}\n\n"
        "Please check that libpymo has been built and is compatible with your "
        "current environment."
    )


libpymo_classes = [
    "ModelOpDefParser",
    "TfEncoding",
    "Quantizer",
    "TensorParams",
    "BlockTensorQuantizer",
]

libpymo_functions = [
    "str_to_dtype",
    "str_to_rank",
    "PtrToInt64",
]


def create_unavailable_class(class_name: str):
    """
    Create unavailable class to lazily throw error when user tries to use the class.
    """

    class _MetaUnavailableClass(type):
        @classmethod
        def __getattr__(mcs, name):
            raise RuntimeError(
                f"Unable to access attribute {name} of class {class_name}: {_error_message()}"
            )

    class _UnavailableClass(metaclass=_MetaUnavailableClass):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"Unable to initialize class {class_name}: {_error_message()}"
            )

        def __getattr__(self, name):
            raise RuntimeError(
                f"Unable to access attribute {name} of class {class_name}: {_error_message()}"
            )

    return type(class_name, (_UnavailableClass,), {})


for libpymo_class in libpymo_classes:
    globals()[libpymo_class] = create_unavailable_class(libpymo_class)


def create_unavailable_function(method_name: str):
    """
    Create unavailable function to lazily throw error when user tries to use the function.
    """

    def unavailable_function(*args, **kwargs):
        raise RuntimeError(f"Unable to run function {method_name}: {_error_message()}")

    return unavailable_function


for libpymo_function in libpymo_functions:
    globals()[libpymo_function] = create_unavailable_function(libpymo_function)


class QuantizationMode(enum.Enum):
    """
    QuantizationMode
    """

    QUANTIZATION_TF = 0
    QUANTIZATION_TF_ENHANCED = 1
    QUANTIZATION_RANGE_LEARNING = 2
    QUANTIZATION_PERCENTILE = 3
    QUANTIZATION_MSE = 4
    QUANTIZATION_ENTROPY = 5


class RoundingMode(enum.Enum):
    """
    RoundingMode
    """

    ROUND_NEAREST = 0
    ROUND_STOCHASTIC = 1


class TensorQuantizerOpMode(enum.Enum):
    """
    TensorQuantizerOpMode
    """

    updateStats = 0
    oneShotQuantizeDequantize = 1
    quantizeDequantize = 2
    passThrough = 3


libpymo_enums = [
    QuantizationMode,
    RoundingMode,
    TensorQuantizerOpMode,
]


for libpymo_enum in libpymo_enums:
    globals().update(libpymo_enum.__members__)
