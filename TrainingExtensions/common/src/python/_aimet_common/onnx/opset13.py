# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=no-member
from typing import Iterable, Optional
from onnx import helper
from . import opset10


class QuantizeLinear(opset10.QuantizeLinear):
    OPSET = 13

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
        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return helper.make_node(
            "QuantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            axis=axis,
        )


class DequantizeLinear(opset10.DequantizeLinear):
    OPSET = 13

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
        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return helper.make_node(
            "DequantizeLinear",
            name=name,
            inputs=list(inputs),
            outputs=[output],
            axis=axis,
        )
