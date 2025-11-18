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
"""Float encoding definition"""

from typing import Union, List, Dict, Optional
import torch
from torch._C._nn import _parse_to as parse_to_args

from aimet_torch.common.defs import EncodingType
from aimet_torch.v2.quantization.base import EncodingBase
from ._finfo import _finfo


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
        maxval: Optional[torch.Tensor],
    ):
        self._finfo = _finfo(exponent_bits, mantissa_bits, finite, unsigned_zero)
        self._maxval = maxval

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
    def maxval(self) -> Optional[torch.Tensor]:
        """
        Returns the maximum representable value of the dequantized tensor
        """
        return self._maxval

    @property
    def bitwidth(self) -> int:
        """
        Returns the bitwidth of the quantizer encoding
        """
        return self.mantissa_bits + self.exponent_bits + 1

    @property
    def granularity(self) -> str:
        """
        Returns the granularity of the quantizer encoding
        """
        if self.maxval is None or self.maxval.dim() == 0:
            return "pertensor"
        non_singleton_dims = tuple(dim for dim in self.maxval.shape if dim > 1)
        if len(non_singleton_dims) <= 1:
            return "perchannel"
        return "unknown"

    def to(self, *args, **kwargs):
        """
        Changes dtype of data in quantizer encoding or device where the data is.
        Behaves similar to torch.Tensor.to
        """
        if self._maxval is None:
            return self

        current_dtype = self._maxval.dtype
        current_device = self._maxval.device

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

        maxval = self._maxval.to(dtype=dtype, device=device)

        return type(self)(
            self.mantissa_bits,
            self.exponent_bits,
            self.finite,
            self.unsigned_zero,
            maxval,
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
            if self.exponent_bits == 5 and self.mantissa_bits == 10:
                # float16
                return {}

            if self.exponent_bits == 8 and self.mantissa_bits == 7:
                # bfloat16
                return {}

            raise NotImplementedError(
                "Floating point encoding export only supports [b]float16; "
                f"got exponent_bits={self.exponent_bits}, mantissa_bits={self.mantissa_bits}"
            )

        raise AssertionError(
            f"Export encoding version {encoding_version} not supported."
        )
