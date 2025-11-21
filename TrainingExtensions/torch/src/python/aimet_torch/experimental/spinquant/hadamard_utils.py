# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Hadamard utilities for SpinQuant"""

from aimet_torch.experimental.spinquant._hadamard_matrices import (
    get_had12,
    get_had20,
    get_had28,
)
import scipy.linalg
import torch

SUPPORTED_FACTORS = {
    12: get_had12,  # Qwen2.5-1.5B (hidden_size=1536), Llama3.2-3B (hidden_size=3072), Phi-3-mini-4k (hidden_size=3072)
    20: get_had20,  # Qwen3 (hidden_size=2560), Qwen2.5-VL ViT (hidden_size=1280)
    28: get_had28,  # Qwen2/2.5-7B (hidden_size=3584)
}
# Powers of two: Llama3.2-1B, phi-1.5 (hidden_size=2048), qwen2.5VL language model (hidden_size=2048)


def is_power_of_two(n: int) -> bool:
    """
    Return True if n is a power of two, False otherwise
    """
    return (n & (n - 1)) == 0 and n > 0


def get_hadamard_matrix(size: int) -> torch.Tensor:
    """
    Get hadamard matrix with dimensions size x size.
    Hadamard matrices with size of powers of two are obtained via scipy.linalg.hadamard.
    For sizes of non powers of two, only sizes which can be decomposed into factor * 2^n for any n>=0 are supported,
    where factor is a key of hadamard_matrices.SUPPORTED_FACTORS.
    Such hadamard matrices are constructed by iteratively taking the Kronecker product of Hadamard size 2 ([1, 1], [1, -1]) starting with Hadamard size 'factor',
    doubling the size of the matrix each iteration until the matrix achieves size 'size'.

    :param size: Size of hadamard matrix to get
    """
    hadamard_matrix = None
    if is_power_of_two(size):
        return torch.tensor(scipy.linalg.hadamard(size), dtype=torch.float)

    had_2 = torch.tensor([[1, 1], [1, -1]])
    for factor, matrix_getter in SUPPORTED_FACTORS.items():
        if size % factor == 0 and is_power_of_two(size // factor):
            hadamard_matrix = matrix_getter()
            while factor != size:
                hadamard_matrix = torch.kron(had_2, hadamard_matrix)
                factor *= 2

    if hadamard_matrix is None:
        raise AssertionError(f"Hadamard matrix of size {size} not supported.")

    return hadamard_matrix
