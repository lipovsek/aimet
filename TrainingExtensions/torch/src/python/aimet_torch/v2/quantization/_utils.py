# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import itertools
from typing import Sequence


def interleave(
    iter1: int | Sequence[int], iter2: int | Sequence[int]
) -> tuple[int, ...]:
    if isinstance(iter1, int):
        if isinstance(iter2, int):
            raise ValueError("At least one of the inputs must be a sequence.")
        iter1 = tuple(itertools.repeat(iter1, len(iter2)))

    if isinstance(iter2, int):
        if isinstance(iter1, int):
            raise ValueError("At least one of the inputs must be a sequence.")
        iter2 = tuple(itertools.repeat(iter2, len(iter1)))

    if len(iter1) != len(iter2):
        raise ValueError("Both sequences must have the same length to interleave.")

    return tuple(itertools.chain.from_iterable(zip(iter1, iter2)))


def concretize_block_size(
    input_shape: Sequence[int], scale_shape: Sequence[int], block_size: Sequence[int]
) -> tuple[int, ...]:
    # Truncate input_shape to match scale_shape length
    input_shape = input_shape[-len(scale_shape) :]

    # Expand block_size to match scale_shape length
    block_size = [
        *(1 for _ in range(len(scale_shape) - len(block_size))),
        *block_size,
    ]

    return tuple(
        block_size[i] if block_size[i] != -1 else input_shape[i] // scale_shape[i]
        for i, _ in enumerate(scale_shape)
    )
