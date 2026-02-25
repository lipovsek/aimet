# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Hadamard matrix utilities - framework-agnostic implementation"""

from .hadamard_matrices import (
    get_had12_data,
    get_had20_data,
    get_had28_data,
)
from .hadamard_utils import (
    is_power_of_two,
    get_hadamard_matrix,
    SUPPORTED_FACTORS,
)

__all__ = [
    "get_had12_data",
    "get_had20_data",
    "get_had28_data",
    "is_power_of_two",
    "get_hadamard_matrix",
    "SUPPORTED_FACTORS",
]
