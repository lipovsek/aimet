# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from packaging.version import parse
from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Optional
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def pack_int8_to_int4x2(arr: np.ndarray) -> np.ndarray:
    if arr.dtype not in (np.int8, np.uint8):
        raise RuntimeError(f"Only [u]int8 can be packed to int4x2; got {arr.dtype}")

    if arr.ndim > 1:
        raise RuntimeError(
            f"Only 1D vector can be packed to int4x2; got N-D array of shape {arr.shape}"
        )

    signed = arr.dtype == np.int8
    arr = arr.astype(np.uint8)

    # Add 0 padding to enable int2x4 packing
    arr = np.concatenate((arr, np.zeros(arr.size % 2, dtype=arr.dtype)))

    if signed:
        # If the int8 value is negative, set the sign bit in the corresponding int4.
        int8_sign_bit = 0b10000000
        int4_sign_bit = 0b00001000
        arr = np.where(arr & int8_sign_bit, arr | int4_sign_bit, arr)

    arr &= 0b00001111
    int4x2 = arr[0::2] << 0 | arr[1::2] << 4
    return int4x2


def pack_int8_to_int2x4(arr: np.ndarray) -> np.ndarray:
    if arr.dtype not in (np.int8, np.uint8):
        raise RuntimeError(f"Only [u]int8 can be packed to int4x2; got {arr.dtype}")

    if arr.ndim > 1:
        raise RuntimeError(
            f"Only 1D vector can be packed to int4x2; got N-D array of shape {arr.shape}"
        )

    signed = arr.dtype == np.int8
    arr = arr.astype(np.uint8)

    # Add 0 padding to enable int2x4 packing
    arr = np.concatenate((arr, np.zeros(4 - (arr.size % 4), dtype=arr.dtype)))

    if signed:
        # If the int8 value is negative, set the sign bit in the corresponding int2.
        int8_sign_bit = 0b10000000
        int2_sign_bit = 0b00000010
        arr = np.where(arr & int8_sign_bit, arr | int2_sign_bit, arr)

    arr &= 0b00000011
    int2x4 = arr[0::4] << 0 | arr[1::4] << 2 | arr[2::4] << 4 | arr[3::4] << 6
    return int2x4


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

        if dtype.startswith("float"):
            return cls._make_float_zeros(zero_point, dtype, name)

        return cls.make_arr(zero_point, dtype, name)

    @classmethod
    def make_arr(cls, arr: np.ndarray, dtype: str, name: str) -> TensorProto:
        cls._check_dtype(dtype)

        if dtype.startswith("float"):
            return cls.make_float_arr(arr, dtype, name)

        return cls.make_int_arr(arr, dtype, name)

    @classmethod
    def make_float_arr(cls, arr: np.ndarray, dtype: str, name: str) -> TensorProto:
        if parse(onnx.__version__) < parse("1.19.0"):
            raise RuntimeError(
                f"{cls.__name__}.make_float_arr requires onnx>=1.19.0; got {onnx.__version__}"
            )

        cls._check_dtype(dtype)

        onnx_dtype = cls.SUPPORTED_DTYPES[dtype]
        np_dtype = helper.tensor_dtype_to_np_dtype(onnx_dtype)
        return numpy_helper.from_array(arr.astype(np_dtype), name=name)

    @classmethod
    def _make_float_zeros(cls, arr: np.ndarray, dtype: str, name: str) -> TensorProto:
        cls._check_dtype(dtype)

        if not dtype.startswith("float4"):
            arr = np.zeros(arr.shape, dtype=np.uint8)
            tensor = numpy_helper.from_array(arr, name=name)
            tensor.data_type = cls.SUPPORTED_DTYPES[dtype]
            return tensor

        target_shape = arr.shape
        arr_float4x2 = pack_int8_to_int4x2(np.zeros(arr.size, dtype=np.uint8))
        tensor = numpy_helper.from_array(arr_float4x2, name=name)
        tensor.data_type = cls.SUPPORTED_DTYPES[dtype]
        tensor.ClearField("dims")
        tensor.dims.extend(target_shape)
        return tensor

    @classmethod
    def make_int_arr(cls, arr: np.ndarray, dtype: str, name: str) -> TensorProto:
        cls._check_dtype(dtype)

        if dtype not in ("int2", "uint2", "int4", "uint4"):
            arr = arr.astype(dtype)
            return numpy_helper.from_array(arr, name=name)

        target_shape = arr.shape
        arr = arr.flatten().astype(np.int8 if dtype in ("int2", "int4") else np.uint8)
        tensor = numpy_helper.from_array(
            pack_int8_to_int4x2(arr)
            if dtype in ("int4", "uint4")
            else pack_int8_to_int2x4(arr),
            name=name,
        )
        # Restore data_type to INT2/INT4
        tensor.data_type = cls.SUPPORTED_DTYPES[dtype]
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
