# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin
"""Float quantizers"""

from contextlib import contextmanager, nullcontext
import functools
from typing import Dict, List, Optional
import math
from packaging.version import parse

import torch
from aimet_torch.quantization.encoding_analyzer import (
    EncodingAnalyzer,
    MinMaxEncodingAnalyzer,
    _flag_extreme_min_max,
)
from aimet_torch.quantization.base import QuantizerBase
from aimet_torch.quantization.float import FloatEncoding
from aimet_torch.quantization.tensor import DequantizedTensor
from aimet_torch.utils import (
    StatisticsNotFoundError,
    patch_attr,
    _is_expandable,
    _torch_compiler_is_exporting,
    _is_qtensor_casting_enabled,
)
from aimet_torch.fp_quantization import fake_cast_to_ieee_float
from ._finfo import _finfo, _torch_dtype_to_finfo, _float4_e2m1fn
from aimet_torch.quantization._utils import blockwise
from ...experimental.onnx import _export as _onnx


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
        exponent_bits (int): Number of exponent bits to simulate. This argument is mutually exclusive with `dtype`.
        mantissa_bits (int):  Number of mantissa bits to simulate. This argument is mutually exclusive with `dtype`.
        finite (bool, optional): If True, +/-inf is representable. Defaults to `False`. Ignored when `dtype` is specified.
        unsigned_zero (bool, optional): If False, +/-0 is representable. Defaults to `True`. Ignored when `dtype` is specified.
        dtype (torch.dtype): torch.dtype to simulate. This argument is mutually exclusive with `exponent_bits` and `mantissa_bits`.
        shape (tuple, optional): Shape of quantization scales. Defaults to `()` (= per-tensor quantization).
        block_size (tuple, optional): If specified, block-wise quantization is performed with the given block size.
        encoding_analyzer (EncodingAnalyzer, optional):
            If specified, quantization scale will be calibrated dynamically based on the input statistics.
            If not specified,
            sub-16-bit floating point quantizers will use min-max encoding analyzer for scale calibration;
            16-bit or higher quantizers will be fixed at scale=1.0

    Examples:

        >>> import aimet_torch.quantization as Q
        >>> input = torch.tensor([[ 1.8998, -0.0947, -1.0891, -0.1727]])
        >>> qdq = Q.float.FloatQuantizeDequantize(dtype=torch.float8_e4m3fnuz)
        >>> with qdq.compute_encodings():
        ...     _ = qdq(input)
        ...
        >>> qdq(input)
        DequantizedTensor([[ 1.8998, -0.0950, -1.1399, -0.1741]])
    """

    maxval: torch.Tensor

    def __init__(
        self,
        exponent_bits: Optional[int] = None,
        mantissa_bits: Optional[int] = None,
        finite: Optional[bool] = None,
        unsigned_zero: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        shape: Optional[tuple[int, ...]] = None,
        block_size: Optional[tuple[int, ...]] = None,
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

        if shape is None:
            self.shape = encoding_analyzer.observer.shape if encoding_analyzer else ()
        else:
            self.shape = shape

        self.block_size = block_size

        if self.bitwidth < 16 and encoding_analyzer is None:
            encoding_analyzer = MinMaxEncodingAnalyzer(self.shape, self.block_size)

        self.encoding_analyzer = encoding_analyzer

        if encoding_analyzer:
            if not _is_expandable(self.encoding_analyzer.observer.shape, self.shape):
                raise RuntimeError(
                    f"Encoding analyzer of shape {self.encoding_analyzer.observer.shape} "
                    f"is incompatible with quantizer of shape {self.shape}."
                )

        maxval = self._finfo.max
        self.register_buffer("maxval", torch.full(self.shape, maxval))

        self._is_overwrite_allowed.update({"maxval": True})

        self._assert_supported_dtype()

    def _assert_supported_dtype(self):
        if self._finfo != _float4_e2m1fn and (
            self._finfo.finite or self._finfo.unsigned_zero
        ):
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
        if torch.onnx.is_in_onnx_export():
            # Bypass get_extra_state during ONNX export.
            # ONNX export doesn't support non-tensor objects in state_dict
            # Return empty tensor since extra state is unnecessary for ONNX export anyway
            return torch.tensor([])

        extra_state_dict = super().get_extra_state()
        finfo = self._finfo
        extra_state_dict.update(
            {
                "exponent_bits": torch.tensor(finfo.exponent_bits),
                "mantissa_bits": torch.tensor(finfo.mantissa_bits),
                "finite": torch.tensor(finfo.finite),
                "unsigned_zero": torch.tensor(finfo.unsigned_zero),
                "block_size": torch.tensor(self.block_size or ()),
            }
        )
        return extra_state_dict

    def set_extra_state(self, state):
        block_size = tuple(state.get("block_size", self.block_size or ()))
        self.block_size = block_size or None

        exponent_bits = state.get("exponent_bits", self._finfo.exponent_bits)
        mantissa_bits = state.get("mantissa_bits", self._finfo.mantissa_bits)
        finite = state.get("finite", self._finfo.finite)
        unsigned_zero = state.get("unsigned_zero", self._finfo.unsigned_zero)

        self._finfo = _finfo(
            exponent_bits=int(exponent_bits),
            mantissa_bits=int(mantissa_bits),
            finite=bool(finite),
            unsigned_zero=bool(unsigned_zero),
        )
        super().set_extra_state(state)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "maxval" in state_dict:
            if self.maxval is None or self.maxval.shape != state_dict["maxval"].shape:
                del self.maxval
                self.register_buffer("maxval", state_dict["maxval"])
        elif self.maxval is not None:
            del self.maxval
            self.register_buffer("maxval", None)

        ret = super().load_state_dict(state_dict, *args, **kwargs)

        if self.maxval is not None:
            self.shape = tuple(self.maxval.shape)

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
                self.get_scale(),
                block_size=self.block_size,
                producer=self,
            )
        return None

    def get_scale(self) -> torch.Tensor:
        log2_scale = self._get_log2_scale()

        if self._finfo == _float4_e2m1fn:
            # For float4_e2m1fn, the scale is restricted to powers of 2, so we round the log2_scale to nearest integer
            log2_scale = torch.round(log2_scale)

        return 2**log2_scale

    def _get_log2_scale(self) -> torch.Tensor:
        return torch.log2(self.maxval.abs()) - math.log2(self._finfo.max)

    @classmethod
    def from_encodings(cls, encodings: FloatEncoding) -> "FloatQuantizeDequantize":
        # pylint: disable=protected-access
        if not isinstance(encodings, FloatEncoding):
            raise TypeError(f"Expected {FloatEncoding}; got {type(encodings)}")

        qtzr = cls(
            *encodings._finfo,
            shape=encodings.scale.shape,
            block_size=encodings.block_size,
        )

        if encodings.scale.numel() == 1 and encodings.scale.item() == 1:
            pass
        else:
            qtzr.maxval = encodings.maxval.clone().detach()

        return qtzr

    @contextmanager
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
        def forward_wrapper(input: torch.Tensor) -> torch.Tensor:
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

        if not _torch_compiler_is_exporting() and type(input) != torch.Tensor:
            input = input.as_subclass(torch.Tensor)

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        if _torch_compiler_is_exporting():
            output = torch.ops.aimet.float_quantize_dequantize(
                input,
                self._finfo,
                encoding.scale,
                encoding.block_size,
            )
        else:
            output = _float_quantize_dequantize(
                input,
                self._finfo,
                encoding.scale,
                encoding.block_size,
            )

        if (
            not _torch_compiler_is_exporting()
            and not torch.onnx.is_in_onnx_export()
            and _is_qtensor_casting_enabled()
        ):
            output = output.as_subclass(DequantizedTensor)
            output.encoding = encoding

        return output

    def extra_repr(self):
        """
        :meta private:
        """
        torch_dtype = self._finfo.to_torch_dtype()

        if torch_dtype is not None:
            extra_repr = [f"dtype={torch_dtype}"]
        else:
            exponent_bits, mantissa_bits, finite, unsigned_zero = self._finfo
            extra_repr = [
                f"exponent_bits={exponent_bits}",
                f"mantissa_bits={mantissa_bits}",
                f"finite={finite}",
                f"unsigned_zero={unsigned_zero}",
            ]

        if self.shape:
            extra_repr.append(f"shape={self.shape}")

        if self.block_size:
            extra_repr.append(f"block_size={self.block_size}")

        return ", ".join(extra_repr)

    @contextmanager
    def _precompute_encodings(self, dtype: torch.dtype | None = None):
        enc = self.get_encodings()

        if enc:
            enc = enc.to(dtype=dtype)
            enc.scale = torch.nn.Parameter(enc.scale, requires_grad=False)

        def get_cache_encodings(*args, **kwargs):  # pylint: disable=unused-argument
            return enc

        def is_initialized(*args, **kwargs):  # pylint: disable=unused-argument
            return enc is not None

        with (
            patch_attr(self, "_parameters", {}) if enc else nullcontext(),
            patch_attr(self, "get_encodings", get_cache_encodings),
            patch_attr(self, "is_initialized", is_initialized),
            patch_attr(self, "scale", enc.scale) if enc else nullcontext(),
        ):
            yield


class QuantizeDequantize(FloatQuantizeDequantize):
    r"""
    Alias of FloatQuantizeDequantize
    """


@_onnx.register_symbolic(_onnx.float_quantize_dequantize_symbolic)
def _float_quantize_dequantize(
    input: torch.Tensor,
    finfo: _finfo,
    scale: torch.Tensor,
    block_size: Optional[tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Fake-cast input to target float dtype.

    Args:
      input: Input tensor
      finfo: Target float dtype
      scale: Scaling factor
    """
    input_q = _float_quantize(input, finfo, scale, block_size)
    return blockwise(
        torch.mul,
        input_q,
        scale,
        block_size=block_size,
    )


if parse(torch.__version__) >= parse("2.4.0"):
    torch.library.define(
        "aimet::float_quantize_dequantize",
        "(Tensor input, int[] finfo, Tensor scale, int[]? block_size) -> Tensor",
    )

    @torch.library.register_fake("aimet::float_quantize_dequantize")
    def _float_quantize_dequantize_meta(  # pylint: disable=unused-argument
        tensor: torch.Tensor,
        finfo: tuple[int, int, bool, bool],
        scale: torch.Tensor,
        block_size: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        return torch.empty_like(tensor)

    @torch.library.impl("aimet::float_quantize_dequantize", "default")
    def _float_quantize_dequantize_impl(
        tensor: torch.Tensor,
        finfo: tuple[int, int, bool, bool],
        scale: torch.Tensor,
        block_size: Optional[tuple[int, ...]] = None,
    ):
        return _float_quantize_dequantize(tensor, _finfo(*finfo), scale, block_size)


def _float_quantize(
    input: torch.Tensor,
    finfo: _finfo,
    scale: torch.Tensor,
    block_size: Optional[tuple[int, ...]] = None,
) -> torch.Tensor:
    if finfo.to_torch_dtype():
        # Well knwon data types. Use cast-decast for better performance
        fake_cast = _cast_decast
    elif not finfo.unsigned_zero:
        # IEEE fake-cast is only valid when unsigned_zero = false
        fake_cast = _fake_cast_to_ieee_float
    else:
        raise NotImplementedError(
            f"Fake-casting to {finfo.to_str()} is not implemented"
        )

    input = blockwise(torch.div, input, scale, block_size=block_size)
    input = input.clamp(-finfo.max, finfo.max)
    return fake_cast(input, finfo)


def _cast_decast(input: torch.Tensor, finfo: _finfo):
    return input.to(finfo.to_torch_dtype()).to(input.dtype)


def _fake_cast_to_ieee_float(input: torch.Tensor, finfo: _finfo):
    return fake_cast_to_ieee_float(
        input, finfo.max, finfo.exponent_bits, finfo.mantissa_bits, finite=finfo.finite
    )
