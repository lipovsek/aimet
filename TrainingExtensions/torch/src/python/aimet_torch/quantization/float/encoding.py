# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin
"""Float encoding definition"""

from typing import Union, List, Dict, TYPE_CHECKING, Optional
import torch
from torch._C._nn import _parse_to as parse_to_args
import onnx

from aimet_torch.common.defs import EncodingType
from aimet_torch.quantization.base import EncodingBase
from ._finfo import _finfo, _float16, _bfloat16, _float4_e2m1fn, _float8_e4m3fn

if TYPE_CHECKING:
    from aimet_torch.quantization.float import FloatQuantizeDequantize


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
        *,
        producer: Optional["FloatQuantizeDequantize"] = None,
    ):
        super().__init__()

        if scale is None:
            raise ValueError("scale cannot be None for FloatEncoding")

        self._finfo = _finfo(exponent_bits, mantissa_bits, finite, unsigned_zero)

        if block_size is not None:
            block_size = tuple(block_size)

        self.scale = scale
        self.block_size = block_size or None
        self.producer = producer

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
    def maxval(self) -> torch.Tensor:
        """
        Returns the maximum representable value of the dequantized tensor
        """
        return self.scale * self._finfo.max

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
        if self.scale is None:
            return self

        current_dtype = self.scale.dtype
        current_device = self.scale.device

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

        scale = self.scale.to(dtype=dtype, device=device)

        return type(self)(
            self.mantissa_bits,
            self.exponent_bits,
            self.finite,
            self.unsigned_zero,
            scale,
            block_size=self.block_size,
        )

    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        from .quantizer import _float_quantize

        if type(input) != torch.Tensor:
            input = input.as_subclass(torch.Tensor)

        return _float_quantize(
            input,
            self._finfo,
            self.scale,
            self.block_size,
        )

    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        from aimet_torch.quantization._utils import blockwise

        if type(input) != torch.Tensor:
            input = input.as_subclass(torch.Tensor)

        return blockwise(
            torch.mul,
            input,
            self.scale,
            block_size=self.block_size,
        )

    def quantize_dequantize(self, input: torch.Tensor) -> torch.Tensor:
        from .quantizer import _float_quantize_dequantize

        if type(input) != torch.Tensor:
            input = input.as_subclass(torch.Tensor)

        return _float_quantize_dequantize(
            input,
            self._finfo,
            self.scale,
            self.block_size,
            getattr(self, "meta_scale", None),
        )

    def to_qnn_encoding_dict(self, encoding_version=None) -> Union[List, Dict]:
        """
        Converts encoding object into QNN encoding
        """
        if encoding_version not in ("2.0.0", "2.1.0") and not (
            torch.all(self.scale == 1) and self._finfo in (_float16, _bfloat16)
        ):
            if self._finfo not in (_float16, _bfloat16):
                reason = f"got dtype={self._finfo.to_str()}"
            else:
                reason = "got non-1 scale"

            raise RuntimeError(
                f"v{encoding_version} floating point encoding only "
                f"supports float16/bfloat16 with scale=1; {reason}"
            )

        if encoding_version == "0.6.1":
            return [{"bitwidth": self.bitwidth, "dtype": "float"}]
        if encoding_version == "1.0.0":
            return {
                "dtype": "FLOAT",
                "bw": self.bitwidth,
                "enc_type": EncodingType.PER_TENSOR.name,
            }

        if encoding_version in ("2.0.0", "2.1.0"):
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

            if self.granularity != "pertensor":
                channel_axis, block_axis = self._safe_get_channel_and_block_axis()

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


class _NVFP4Encoding(FloatEncoding):
    """
    Encoding object for NVidia FP4 quantization
    """

    def __init__(
        self,
        scale: torch.Tensor,
        meta_scale: torch.Tensor,
        block_size: tuple[int, ...],
        *,
        producer: Optional["FloatQuantizeDequantize"] = None,
    ):
        self.meta_scale = meta_scale

        super().__init__(
            mantissa_bits=_float4_e2m1fn.mantissa_bits,
            exponent_bits=_float4_e2m1fn.exponent_bits,
            finite=_float4_e2m1fn.finite,
            unsigned_zero=_float4_e2m1fn.unsigned_zero,
            scale=scale,
            block_size=block_size,
            producer=producer,
        )

    def to_qnn_encoding_dict(self, encoding_version=None) -> Dict:
        from .quantizer import _float_quantize

        if encoding_version != "2.1.0":
            raise RuntimeError(
                f"NVFP4 encoding is only supported in 2.1.0 encoding; got {encoding_version}"
            )

        encoding_dict = super().to_qnn_encoding_dict(encoding_version)
        quantized_scale = _float_quantize(
            self.scale,
            _float8_e4m3fn,
            self.meta_scale,
        )
        encoding_dict.update(
            {
                "y_scale": {
                    "x": quantized_scale.tolist(),
                    "x_scale": self.meta_scale.tolist(),
                    "input_dtype": "float8e4m3fn",
                }
            }
        )
        return encoding_dict

    @classmethod
    def _from_float_encoding(
        cls, encoding: FloatEncoding, meta_scale: torch.Tensor
    ) -> "_NVFP4Encoding":
        """
        Create an NVFP4 encoding from a FloatEncoding and a meta scale
        """
        # pylint: disable=protected-access
        if not isinstance(encoding, FloatEncoding) or isinstance(
            encoding, _NVFP4Encoding
        ):
            raise TypeError("Cannot create NVFP4 encoding from another NVFP4 encoding")

        if encoding.block_size is None:
            raise ValueError(
                "Cannot create NVFP4 encoding from a FloatEncoding with no block size"
            )

        if encoding._finfo != _float4_e2m1fn:
            raise ValueError(
                f"Cannot create NVFP4 encoding from {encoding._finfo.to_str()} encoding"
            )

        from .quantizer import _float_quantize_dequantize

        qdq_scale = _float_quantize_dequantize(
            encoding.scale,
            _float8_e4m3fn,
            meta_scale,
        )

        if not torch.allclose(encoding.scale, qdq_scale):
            raise ValueError(
                "Cannot create NVFP4 encoding from a FloatEncoding with a scale "
                "that cannot be represented in float8e4m3fn"
            )

        return cls(
            scale=encoding.scale,
            meta_scale=meta_scale,
            block_size=encoding.block_size,
            producer=encoding.producer,
        )
