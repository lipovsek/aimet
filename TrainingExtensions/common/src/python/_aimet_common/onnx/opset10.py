# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
# pylint: disable=no-member
from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Optional
import numpy as np
from onnx import helper, numpy_helper, TensorProto


def pack_int8_to_int4x2(arr: np.ndarray) -> np.ndarray:
    if arr.dtype not in (np.int8, np.uint8):
        raise RuntimeError(f"Only [u]int8 can be packed to int4x2; got {arr.dtype}")

    if arr.ndim > 1:
        raise RuntimeError(
            f"Only 1D vector can be packed to int4x2; got N-D array of shape {arr.shape}"
        )

    if arr.size % 2 == 1:
        # Add 0 padding to enable int4x2 packing
        arr = np.concatenate((arr, np.array([0], dtype=arr.dtype)))

    signed = arr.dtype == np.int8
    arr = arr.astype(np.uint8)

    int4x2 = np.zeros(arr.size // 2, dtype=np.uint8)
    int4x2 |= arr[1::2] << 4

    if signed:
        int4x2 |= arr[::2] & 0x07
        int4x2 |= (arr[::2] & 0x80) >> 4
    else:
        int4x2 |= arr[::2] & 0x0F

    return int4x2


def unpack_int4x2_to_int8(arr: np.ndarray, dtype) -> np.ndarray:
    if arr.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 input; got {arr.dtype}")

    dtype = np.dtype(dtype)
    if dtype not in (np.int8, np.uint8):
        raise RuntimeError(f"Expected target dtype [u]int8; got {dtype}")

    if arr.ndim > 1:
        raise RuntimeError(
            f"Only 1D vector can be packed to int4x2; got N-D array of shape {arr.shape}"
        )

    uint8 = np.empty(arr.size * 2, dtype=np.uint8)
    uint8[1::2] = arr >> 4
    uint8[0::2] = arr & 0x0F

    if dtype == np.uint8:
        return uint8

    int8 = np.where(uint8 >= 8, uint8 | 0xF0, uint8).astype(np.int8)
    return int8


class _QdqNodeFactory(ABC):
    OPSET: int
    SUPPORTED_DTYPES: Mapping[str, "TensorProto.DataType"]

    @classmethod
    @abstractmethod
    def make_node(
        cls,
        name: str,
        inputs: Iterable[str],
        output: str,
        dtype: str,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
    ): ...

    @classmethod
    def _check_dtype(cls, dtype: str):
        if dtype in cls.SUPPORTED_DTYPES:
            return

        raise RuntimeError(
            f"Unsupported dtype {dtype}; "
            f"opset {cls.OPSET} expects one of {list(cls.SUPPORTED_DTYPES.keys())}"
        )

    @classmethod
    def make_zero_point(
        cls, zero_point: np.ndarray, dtype: str, name: str
    ) -> TensorProto:
        cls._check_dtype(dtype)

        if (dtype == "int32" or dtype.startswith("float")) and not np.all(
            zero_point == 0
        ):
            raise RuntimeError(
                "DequantizeLinear with type int32 or float8 should have "
                "no zero point or all zero points should be 0"
            )
        return cls.make_int_arr(zero_point, dtype, name)

    @classmethod
    def make_int_arr(cls, arr: np.ndarray, dtype: str, name: str) -> TensorProto:
        if dtype not in ("int4", "uint4"):
            arr = arr.astype(dtype)
            return numpy_helper.from_array(arr, name=name)

        target_shape = arr.shape
        arr_int4x2 = pack_int8_to_int4x2(
            arr.flatten().astype(np.int8 if dtype == "int4" else np.uint8)
        )
        tensor = numpy_helper.from_array(arr_int4x2, name=name)

        # Restore data_type to INT4/UINT4
        tensor.data_type = TensorProto.INT4 if dtype == "int4" else TensorProto.UINT4
        tensor.ClearField("dims")
        tensor.dims.extend(target_shape)

        return tensor


class QuantizeLinear(_QdqNodeFactory):
    OPSET = 10
    SUPPORTED_DTYPES = {
        "int8": TensorProto.INT8,
        "uint8": TensorProto.UINT8,
    }

    @classmethod
    def make_node(
        cls,
        name: str,
        inputs: Iterable[str],
        output: str,
        dtype: str,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        if axis is not None:
            raise RuntimeError(
                f"Per-channel quantization is not supported in opset {cls.OPSET}"
            )

        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return helper.make_node(
            "QuantizeLinear", name=name, inputs=list(inputs), outputs=[output]
        )


class DequantizeLinear(_QdqNodeFactory):
    OPSET = 10
    SUPPORTED_DTYPES = {
        "int8": TensorProto.INT8,
        "uint8": TensorProto.UINT8,
        "int32": TensorProto.INT32,
    }

    @classmethod
    def make_node(
        cls,
        name: str,
        inputs: Iterable[str],
        output: str,
        dtype: str,
        axis: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        if axis is not None:
            raise RuntimeError(
                f"Per-channel quantization is not supported in opset {cls.OPSET}"
            )

        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return helper.make_node(
            "DequantizeLinear", name=name, inputs=list(inputs), outputs=[output]
        )
