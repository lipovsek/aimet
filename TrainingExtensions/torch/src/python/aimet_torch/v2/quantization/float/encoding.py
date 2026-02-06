# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin
"""Float encoding definition"""

from typing import Union, List, Dict
import torch
from torch._C._nn import _parse_to as parse_to_args
import onnx

from aimet_torch.common.defs import EncodingType
from aimet_torch.v2.quantization.base import EncodingBase
from ._finfo import _finfo, _float16, _bfloat16


__all__ = ["FloatEncoding"]


class FloatEncoding(EncodingBase):
    """
    Encoding object for float quantization
    """

    def __init__(
        self,
        mantissa_bits: int,
        exponent_bits: int,
        finite: bool,
        unsigned_zero: bool,
        scale: torch.Tensor,
        block_size: tuple[int, ...] | None = None,
    ):
        if scale is None:
            raise ValueError("scale cannot be None for FloatEncoding")

        self._finfo = _finfo(exponent_bits, mantissa_bits, finite, unsigned_zero)

        if block_size is not None:
            block_size = tuple(block_size)

        self._scale = scale
        self._block_size = block_size or None

    @property
    def mapping(self) -> str:
        """
        Returns the mapping method for this encoding
        """
        return "float"

    @property
    def mantissa_bits(self) -> int:
        """
        Return number of mantissa bits in float representation
        """
        return self._finfo.mantissa_bits

    @property
    def exponent_bits(self) -> int:
        """
        Returns the number of exponent bits in float representation
        """
        return self._finfo.exponent_bits

    @property
    def finite(self) -> bool:
        """
        Returns True if +/-inf is representable
        """
        return self._finfo.finite

    @property
    def unsigned_zero(self) -> bool:
        """
        Returns True if -0 or -nan is NOT representable
        """
        return self._finfo.unsigned_zero

    @property
    def scale(self) -> torch.Tensor:
        """
        Returns the scale of the quantizer encoding
        """
        return self._scale

    @property
    def maxval(self) -> torch.Tensor:
        """
        Returns the maximum representable value of the dequantized tensor
        """
        return self._scale * self._finfo.max

    @property
    def block_size(self) -> tuple[int, ...] | None:
        """
        Returns the block size for block floating point quantization
        """
        return self._block_size

    @property
    def bitwidth(self) -> int:
        """
        Returns the bitwidth of the quantizer encoding
        """
        return self.mantissa_bits + self.exponent_bits + 1

    def to(self, *args, **kwargs):
        """
        Changes dtype of data in quantizer encoding or device where the data is.
        Behaves similar to torch.Tensor.to
        """
        if self._scale is None:
            return self

        current_dtype = self._scale.dtype
        current_device = self._scale.device

        to_args = parse_to_args(*args, **kwargs)
        device, dtype, _, _ = to_args

        dtype = dtype or current_dtype
        device = device or current_device

        if dtype == current_dtype and device == current_device:
            return self

        if dtype and not dtype.is_floating_point:
            raise RuntimeError(
                f"Cannot change encoding data dtype to {dtype}, "
                "only floating point data types are supported"
            )

        scale = self._scale.to(dtype=dtype, device=device)

        return type(self)(
            self.mantissa_bits,
            self.exponent_bits,
            self.finite,
            self.unsigned_zero,
            scale,
        )

    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_qnn_encoding_dict(self, encoding_version=None) -> Union[List, Dict]:
        """
        Converts encoding object into QNN encoding
        """
        if encoding_version == "0.6.1":
            return [{"bitwidth": self.bitwidth, "dtype": "float"}]
        if encoding_version == "1.0.0":
            return {
                "dtype": "FLOAT",
                "bw": self.bitwidth,
                "enc_type": EncodingType.PER_TENSOR.name,
            }

        if encoding_version == "2.0.0":
            if self._finfo in (_float16, _bfloat16):
                # v2 encoding doesn't treat float16/bfloat16 as quantized dtypes
                return {}

            onnx_dtype = self._finfo.to_onnx_dtype()

            if onnx_dtype is None:
                raise RuntimeError

            y_scale = self.scale
            onnx_dtype_str = onnx.helper.tensor_dtype_to_string(onnx_dtype)
            _, onnx_dtype_str = onnx_dtype_str.lower().split(".")

            channel_axis = None
            block_axis = None
            block_size = None

            if self.granularity == "pertensor":
                pass
            elif self.granularity == "perchannel":
                channel_axis = self._get_channel_axis()
            elif self.granularity == "blockwise":
                # NOTE: This sometimes fail
                block_axis = self._get_block_axis()
            else:
                raise NotImplementedError

            if block_axis is not None:
                axis = block_axis
                block_size = self.block_size[block_axis]
            elif channel_axis is not None:
                axis = channel_axis
                y_scale = y_scale.flatten()
            else:
                axis = None
                y_scale = y_scale.squeeze()

            y_scale = y_scale.tolist()

            ret = {
                "output_dtype": onnx_dtype_str,
                "y_scale": y_scale,
            }
            if axis is not None:
                ret.update({"axis": axis})
            if block_size is not None:
                ret.update({"block_size": block_size})

            return ret

        raise AssertionError(
            f"Export encoding version {encoding_version} not supported."
        )
