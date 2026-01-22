# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from typing import Iterable, Optional
from onnx import helper, TensorProto
from ..onnx import opset13


class QuantizeLinear(opset13.QuantizeLinear):
    OPSET = 21
    SUPPORTED_DTYPES = {
        **opset13.QuantizeLinear.SUPPORTED_DTYPES,
        "int4": TensorProto.INT4,
        "uint4": TensorProto.UINT4,
        "int16": TensorProto.INT16,
        "uint16": TensorProto.UINT16,
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
        cls._check_dtype(dtype)

        if axis is None and block_size is not None:
            raise RuntimeError(
                "axis must be specified if block_size is not None; "
                f"got axis={axis}, block_size={block_size}"
            )

        return helper.make_node(
            "QuantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            # NOTE: Don't pass output_dtype explicitly; ORT has a bug
            #       where per-tensor int8 QuantizeLinear
            #       fails with output_dtype explicitly specified as INT8
            # output_dtype=cls.SUPPORTED_DTYPES[dtype],
            axis=axis,
            block_size=block_size,
        )


class DequantizeLinear(opset13.DequantizeLinear):
    OPSET = 21
    SUPPORTED_DTYPES = {
        **opset13.DequantizeLinear.SUPPORTED_DTYPES,
        "int4": TensorProto.INT4,
        "uint4": TensorProto.UINT4,
        "int16": TensorProto.INT16,
        "uint16": TensorProto.UINT16,
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
        cls._check_dtype(dtype)

        if axis is None and block_size is not None:
            raise RuntimeError(
                "axis must be specified if block_size is not None; "
                f"got axis={axis}, block_size={block_size}"
            )

        return helper.make_node(
            "DequantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            axis=axis,
            block_size=block_size,
        )
