# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Definitions for ONNX"""

import abc
from typing import Union, Optional
from dataclasses import dataclass
import numpy as np

from aimet_onnx.common.defs import qtype, int2, Int


class DataLoader:
    """
    Example of a Dataloader which can be used for running AMPv2
    """

    def __init__(self, data: np.ndarray, batch_size: int, iterations: int):
        """
        :param data: Numpy array
        :param batch_size: batch size for data loader
        :param iterations: number of iterations
        """
        self._data = data
        self.batch_size = batch_size
        self.iterations = iterations

    def __iter__(self):
        """Iterates over dataset"""

    def __len__(self):
        """Returns number of batches the dataloader will iterate"""
        return self.iterations


class Granularity(abc.ABC):
    """Parent class describing granularity of quantization encodings"""


@dataclass(frozen=True)
class PerTensor(Granularity):
    """A single set of quantization parameters is shared across the entire tensor"""


@dataclass(frozen=True)
class PerChannel(Granularity):
    """One set of independent quantization parameters per output channel"""


@dataclass(frozen=True)
class Blockwise(Granularity):
    """One set of independent quantization parameters per block"""

    block_size: int


@dataclass(frozen=True)
class LPBQ(Blockwise):
    """Low-power blockwise quantization: blockwise scales are quantized to an integer grid"""

    block_size: int = 64
    scale_bits: int = 4


@dataclass(frozen=True)
class QSpec:
    """
    Specifies how a tensor should be quantized (precision and granularity).

    Args:
        dtype (qtype): Quantized data type
        granularity (Granularity | None): How quantization parameters are shared across the tensor
        symmetric (bool | None): If specified, determines whether encodings will be symmetric
        shift_zero_point (bool): Whether to shift quantizer's 0 point by a half-step.
            Only supported for int2 param quantizers
    """

    dtype: qtype
    granularity: Optional[Granularity] = None
    symmetric: Optional[bool] = None
    shift_zero_point: bool = False

    def __post_init__(self):
        if self.shift_zero_point:
            if not self.symmetric:
                raise ValueError(
                    "Zero point shift only supported for symmetric quantization"
                )
            if self.dtype != int2:
                raise ValueError(
                    "Zero point shift only supported for int2 quantization"
                )
        if isinstance(self.granularity, LPBQ):
            if not isinstance(self.dtype, Int):
                raise ValueError("LPBQ is only supported for integer quantization")
            if not self.symmetric:
                raise ValueError("LPBQ is only supported for symmetric quantization")

    @classmethod
    def per_tensor(
        cls,
        dtype: Union[qtype, str],
        *,
        symmetric: Optional[bool] = None,
        shift_zero_point: bool = False,
    ) -> "QSpec":
        """Constructs a per-tensor QSpec"""
        return cls(qtype.as_qtype(dtype), PerTensor(), symmetric, shift_zero_point)

    @classmethod
    def per_channel(
        cls,
        dtype: Union[qtype, str],
        *,
        symmetric: Optional[bool] = None,
        shift_zero_point: bool = False,
    ) -> "QSpec":
        """Constructs a per-channel QSpec"""
        return cls(qtype.as_qtype(dtype), PerChannel(), symmetric, shift_zero_point)

    @classmethod
    def blockwise(
        cls,
        dtype: Union[qtype, str],
        block_size: int,
        *,
        symmetric: Optional[bool] = None,
        shift_zero_point: bool = False,
    ) -> "QSpec":
        """Constructs a blockwise QSpec"""
        return cls(
            qtype.as_qtype(dtype), Blockwise(block_size), symmetric, shift_zero_point
        )

    @classmethod
    def lpbq(
        cls,
        dtype: Union[qtype, str],
        block_size: int = 64,
        scale_bits: int = 4,
    ) -> "QSpec":
        """Constructs a low-power blockwise QSpec"""
        return cls(qtype.as_qtype(dtype), LPBQ(block_size, scale_bits), symmetric=True)
