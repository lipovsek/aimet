# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=redefined-builtin
"""Float quantizers"""

import contextlib
import functools
from typing import Dict, List, Optional
import math

import torch
from aimet_torch.v2.quantization.encoding_analyzer import (
    EncodingAnalyzer,
    _flag_extreme_min_max,
)
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.float import FloatEncoding
from aimet_torch.v2.quantization.tensor import DequantizedTensor
from aimet_torch.v2.utils import StatisticsNotFoundError, patch_attr
from aimet_torch.fp_quantization import fake_cast_to_ieee_float
from ._finfo import _finfo, _torch_dtype_to_finfo


__all__ = ["QuantizeDequantize", "FloatQuantizeDequantize"]


class FloatQuantizeDequantize(QuantizerBase):  # pylint: disable=abstract-method
    r"""
    Simulates quantization by fake-casting the input

    If dtype is provided, this is equivalent to

    .. math::
        out = x.to(dtype).to(x.dtype) \\


    If the exponent and mantissa bits are provided, this is equivalent to

    .. math::
        out = \left\lceil\frac{x_c}{scale}\right\rfloor * scale

    where

    .. math::
        x_c   &= clamp(x, -max, max) \\
        bias  &= 2^{exponent} - \log_2(max) + \log_2(2 - 2^{-mantissa}) - 1 \\
        scale &= 2 ^ {\left\lfloor \log_2 |x_c| + bias \right\rfloor - mantissa - bias} \\


    The IEEE standard computes the maximum representable value by

    .. math::
        max = (2 - 2^{-mantissa}) * 2^{(\left\lfloor 0.5 * exponent\_max \right\rfloor)} \\

    where

    .. math::
        exponent\_max = 2^{exponent} - 1 \\

    Args:
        exponent_bits (int): Number of exponent bits to simulate
        mantissa_bits (int):  Number of mantissa bits to simulate
        dtype (torch.dtype): torch.dtype to simulate. This argument is mutually exclusive with exponent_bits and mantissa_bits.
        encoding_analyzer (EncodingAnalyzer): If specified, the maximum value to represent will be determined dynamically based on the input statistics for finer precision.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.tensor([[ 1.8998, -0.0947],[-1.0891, -0.1727]])
        >>> qdq = Q.float.FloatQuantizeDequantize(mantissa_bits=7, exponent_bits=8)
        >>> # Unlike AffineQuantizer, FloatQuantizer is initialized without calling compute_encodings()
        >>> qdq.is_initialized()
        True
        >>> qdq.is_bfloat16()
        True
        >>> qdq.bitwidth
        16
        >>> qdq(input)
        tensor([[ 1.8984, -0.0947], [-1.0859, -0.1729]])

        >>> from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
        >>> encoding_analyzer = MinMaxEncodingAnalyzer(shape=[])
        >>> qdq = Q.float.FloatQuantizeDequantize(dtype=torch.float16, encoding_analyzer=encoding_analyzer)
        >>> qdq.is_float16()
        True
        >>> qdq.bitwidth
        16
        >>> qdq(input)
        tensor([[ 1.8994, -0.0947], [-1.0889, -0.1727]])
    """

    maxval: Optional[torch.Tensor]

    def __init__(
        self,
        exponent_bits: Optional[int] = None,
        mantissa_bits: Optional[int] = None,
        finite: Optional[bool] = None,
        unsigned_zero: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        encoding_analyzer: Optional[EncodingAnalyzer] = None,
    ):
        super().__init__()

        if dtype is None:
            if exponent_bits is None or mantissa_bits is None:
                raise ValueError(
                    'Neither "dtype" nor "exponent/mantissa_bits" was specified.'
                )

            if finite is None:
                finite = False

            if unsigned_zero is None:
                unsigned_zero = False

        if dtype is not None:
            if (
                exponent_bits is not None
                or mantissa_bits is not None
                or finite is not None
                or unsigned_zero is not None
            ):
                raise ValueError(
                    'Argument "dtype" is mutually exclusive with "exponent/mantissa_bits/finite/unsigned_zero".'
                )

            exponent_bits, mantissa_bits, finite, unsigned_zero = (
                _finfo.from_torch_dtype(dtype)
            )

        self._finfo = _finfo(exponent_bits, mantissa_bits, finite, unsigned_zero)

        self.encoding_analyzer = encoding_analyzer

        if self.encoding_analyzer:
            shape = self.encoding_analyzer.observer.shape
            maxval = self._finfo.max
            self.register_buffer("maxval", torch.full(shape, maxval))
        else:
            self.register_buffer("maxval", None)

        self._is_overwrite_allowed.update({"maxval": True})

        self._assert_supported_dtype()

    def _assert_supported_dtype(self):
        if self._finfo.finite or self._finfo.unsigned_zero:
            if self._finfo.to_torch_dtype() is None:
                torch_special_builtin_dtypes = [
                    dtype
                    for dtype in _torch_dtype_to_finfo
                    if dtype not in (torch.float16, torch.bfloat16)
                ]
                msg = " ".join(
                    [
                        "finite/unsigned_zero floating point has limited support.",
                        f"Expected PyTorch built-in data types, such as {torch_special_builtin_dtypes};",
                        f"got '{self._finfo.to_str()}'",
                    ]
                )
                raise RuntimeError(msg)

    @property
    def exponent_bits(self):
        """Returns exponent bits"""
        return self._finfo.exponent_bits

    @exponent_bits.setter
    def exponent_bits(self, exponent_bits: int):
        _, mantissa_bits, finite, unsigned_zero = self._finfo
        self._finfo = _finfo(exponent_bits, mantissa_bits, finite, unsigned_zero)

    @property
    def mantissa_bits(self):
        """Returns mantissa bits"""
        return self._finfo.mantissa_bits

    @mantissa_bits.setter
    def mantissa_bits(self, mantissa_bits: int):
        exponent_bits, _, finite, unsigned_zero = self._finfo
        self._finfo = _finfo(exponent_bits, mantissa_bits, finite, unsigned_zero)

    def get_extra_state(self):
        extra_state_dict = super().get_extra_state()
        extra_state_dict["exponent_bits"] = torch.tensor(self.exponent_bits)
        extra_state_dict["mantissa_bits"] = torch.tensor(self.mantissa_bits)
        return extra_state_dict

    def set_extra_state(self, state):
        self.exponent_bits = state["exponent_bits"].item()
        self.mantissa_bits = state["mantissa_bits"].item()
        super().set_extra_state(state)

    def load_state_dict(self, state_dict, strict: bool = True):
        if "maxval" in state_dict:
            if self.maxval is None:
                del self.maxval
                self.register_buffer("maxval", state_dict["maxval"])
        elif self.maxval is not None:
            del self.maxval
            self.register_buffer("maxval", None)

        ret = super().load_state_dict(state_dict, strict)
        return ret

    @property
    def bitwidth(self):
        """
        Returns bitwidth of the quantizer
        """
        return self.exponent_bits + self.mantissa_bits + 1

    def is_float16(self):
        """
        Returns true if current configuration simulates IEEE float16
        """
        return self._finfo.is_float16()

    def is_bfloat16(self):
        """
        Returns true if current configuration simulates bfloat16
        """
        return self._finfo.is_bfloat16()

    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        :meta private:
        """
        return [{"bitwidth": self.bitwidth, "dtype": "float"}]

    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        :meta private:
        Set encodings represented in the same format as the output of get_legacy_encodings as below:

        [
            {'bitwidth': int, 'dtype': str},
            ...
        ]
        """
        if encodings[0]["bitwidth"] != 16:
            raise RuntimeError(
                f"{self.__class__} can only import 16-bit legay encodings."
            )
        self.exponent_bits = 5
        self.mantissa_bits = 10

    def get_encodings(self) -> Optional[FloatEncoding]:
        if self.is_initialized():
            return FloatEncoding(
                self._finfo.mantissa_bits,
                self._finfo.exponent_bits,
                self._finfo.finite,
                self._finfo.unsigned_zero,
                self.maxval,
            )
        return None

    def get_scale(self) -> Optional[torch.Tensor]:
        log2_scale = self._get_log2_scale()

        if log2_scale is None:
            return None

        return 2**log2_scale

    def _get_log2_scale(self) -> Optional[torch.Tensor]:
        if self.maxval is None:
            return None

        return torch.log2(self.maxval.abs()) - math.log2(self._finfo.max)

    @classmethod
    def from_encodings(cls, encodings: FloatEncoding) -> "FloatQuantizeDequantize":
        if not isinstance(encodings, FloatEncoding):
            raise TypeError(f"Expected {FloatEncoding}; got {type(encodings)}")

        qtzr = cls(
            exponent_bits=encodings.exponent_bits, mantissa_bits=encodings.mantissa_bits
        )

        if encodings.maxval is not None:
            qtzr.maxval.copy_(encodings.maxval)

        return qtzr

    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        During ``compute_encodings`` is enabled, the quantizer forward pass performs
        dynamic quantization using the batch statistics.
        """
        if not self.encoding_analyzer or not any(self._is_overwrite_allowed.values()):
            yield
            return

        original_forward = self.forward

        @functools.wraps(original_forward)
        def forward_wrapper(input):
            input = input.as_subclass(torch.Tensor)
            batch_statistics = self.encoding_analyzer.update_stats(input)
            num_steps = math.pow(2, self.bitwidth) - 1
            dynamic_min, dynamic_max = (
                self.encoding_analyzer.compute_encodings_from_stats(
                    batch_statistics, num_steps, is_symmetric=False
                )
            )
            dynamic_absmax = torch.maximum(dynamic_min.abs(), dynamic_max.abs())
            dynamic_absmax = dynamic_absmax.to(
                dtype=self.maxval.dtype, device=self.maxval.device
            ).expand_as(self.maxval)

            with patch_attr(self, "maxval", dynamic_absmax):
                return original_forward(input)

        self.encoding_analyzer.reset_stats()

        try:
            with patch_attr(self, "forward", forward_wrapper):
                yield
        except:  # pylint: disable=try-except-raise
            raise

        try:
            num_steps = math.pow(2, self.bitwidth) - 1
            min, max = self.encoding_analyzer.compute_encodings(
                num_steps, is_symmetric=False
            )
            _flag_extreme_min_max(min, max)
        except StatisticsNotFoundError:
            return

        if min is None or max is None:
            return

        absmax = torch.maximum(min.abs(), max.abs()).expand_as(self.maxval)
        absmax = absmax.to(dtype=self.maxval.dtype, device=self.maxval.device)
        with torch.no_grad():
            self.maxval.copy_(absmax)

    def forward(self, input: torch.Tensor):
        """
        :param input: Input to quantize and dequantize
        :return: Quantize-dequantized output
        """
        if not input.is_floating_point():
            return input

        self._assert_supported_dtype()

        if not self.is_initialized():
            raise RuntimeError(
                "Failed to run FloatQuantizeDequantize since quantization parameters are not initialized."
                " Please initialize the quantization parameters using `compute_encodings()`."
            )

        encoding = self.get_encodings()
        assert encoding is not None

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        output = _fake_cast(
            input.as_subclass(torch.Tensor), self._finfo, self.get_scale()
        )
        output = output.as_subclass(DequantizedTensor)
        output.encoding = encoding
        return output

    def extra_repr(self):
        """
        :meta private:
        """
        if self.maxval is None:
            torch_dtype = self._finfo.to_torch_dtype()

            if torch_dtype is not None:
                return f"dtype={torch_dtype}"

        exponent_bits, mantissa_bits, finite, unsigned_zero = self._finfo

        return " ".join(
            [
                f"exponent_bits={exponent_bits}",
                f"mantissa_bits={mantissa_bits}",
                f"finite={finite}",
                f"unsigned_zero={unsigned_zero}",
            ]
        )


class QuantizeDequantize(FloatQuantizeDequantize):
    r"""
    Alias of FloatQuantizeDequantize
    """


def _fake_cast(
    input: torch.Tensor,
    finfo: _finfo,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fake-cast input to target float dtype.

    Args:
      input: Input tensor
      finfo: Target float dtype
      scale: Scaling factor
    """
    if finfo.to_torch_dtype():
        # Well knwon data types. Use cast-decast for better performance
        fake_cast = _cast_decast
    elif not finfo.finite and not finfo.unsigned_zero:
        # IEEE fake-cast is only valid when finite = unsigned_zero = false
        fake_cast = _fake_cast_to_ieee_float
    else:
        raise NotImplementedError(
            f"Fake-casting to {finfo.to_str()} is not implemented"
        )

    # Analogous to quantize
    if scale is not None:
        input = input / scale
    input = input.clamp(-finfo.max, finfo.max)
    input = fake_cast(input, finfo)

    # Analogous to dequantize
    if scale is not None:
        input = input * scale

    return input


def _cast_decast(input: torch.Tensor, finfo: _finfo):
    return input.to(finfo.to_torch_dtype()).to(input.dtype)


def _fake_cast_to_ieee_float(input: torch.Tensor, finfo: _finfo):
    return fake_cast_to_ieee_float(
        input, finfo.max, finfo.exponent_bits, finfo.mantissa_bits
    )
