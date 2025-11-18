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
