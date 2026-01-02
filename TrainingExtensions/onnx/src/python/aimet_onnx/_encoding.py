# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=redefined-builtin
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import numpy as np
from typing import Any, Literal, TypeVar, Type, TYPE_CHECKING
from aimet_onnx.common.defs import EncodingType, QuantizationDataType
from aimet_onnx.common import libpymo

from . import lpbq_utils

if TYPE_CHECKING:
    from aimet_onnx.qc_quantize_op import QcQuantizeOp, GroupedBlockQuantizeDequantize


T = TypeVar("T", bound="EncodingBase")


class EncodingBase(ABC):
    @abstractmethod
    def to_qnn_encoding_dict(
        self, encoding_version: str | None = None
    ) -> list | dict[str, Any]:
        """
        Convert EncodingBase object to QNN encoding dict format.

        Args:
          encoding_version: Version of QNN encoding format
        """

    @classmethod
    @abstractmethod
    def from_qnn_encoding_dict(
        cls: Type[T],
        encoding_dict: list | dict[str, Any],
        input_shape: tuple[int, ...] | None = None,
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,
    ) -> T:
        """
        Create EncodingBase object from QNN encoding dict format.

        Args:
            encoding_dict: QNN encoding dict
            input_shape (optional):
                Input shape of the tensor being quantized.
                Used to infer encoding shape for blockwise quantization.
                Only required for 1.0.0 BQ encoding; ignored in all other cases
            default_channel_axis (optional):
                Default channel axis to use if channel axis isn't specified in encoding_dict.
                Required for 1.0.0 BQ encoding;
                optional but recommended for 2.0.0 BQ encoding;
                ignored in all other cases
            default_block_axis (optional):
                Default block axis to use if block axis isn't specified in encoding_dict.
                Only required for 1.0.0 BQ encoding; ignored in all other cases
        """
        subcls = cls.get_subclass(encoding_dict)

        return subcls.from_qnn_encoding_dict(
            encoding_dict,
            input_shape=input_shape,
            default_channel_axis=default_channel_axis,
            default_block_axis=default_block_axis,
        )

    @classmethod
    def get_subclass(cls, encoding_dict) -> Type[EncodingBase]:
        version = cls._infer_encoding_version(encoding_dict)

        if version == "0.6.1":
            subcls = (
                AffineEncoding if encoding_dict[0]["dtype"] == "int" else FloatEncoding
            )
        elif version == "1.0.0":
            if encoding_dict["enc_type"] == EncodingType.LPBQ.name:
                subcls = LPBQEncoding
            elif encoding_dict["dtype"] == "INT":
                subcls = AffineEncoding
            else:
                subcls = FloatEncoding
        else:
            if "per_block_int_scale" in encoding_dict:
                subcls = LPBQEncoding
            elif "int" in encoding_dict["output_dtype"]:
                subcls = AffineEncoding
            else:
                subcls = FloatEncoding

        return subcls

    @abstractmethod
    def load_to(self, qtzr: QcQuantizeOp) -> None:
        """
        Load encoding to QcQuantizeOp object
        """

    @classmethod
    def _infer_encoding_version(cls, encoding_dict: list | dict[str, Any]) -> str:
        if isinstance(encoding_dict, list):
            version = "0.6.1"
        else:
            version = "1.0.0" if "bw" in encoding_dict else "2.0.0"

        return version

    @classmethod
    @abstractmethod
    def from_quantizer(cls: Type[T], qtzr: QcQuantizeOp) -> T | None:
        """
        Create EncodingBase object from QcQuantizeOp object.

        Args:
            qtzr: QcQuantizeOp object
        """
        from aimet_onnx.qc_quantize_op import GroupedBlockQuantizeDequantize

        if (
            isinstance(qtzr, GroupedBlockQuantizeDequantize)
            and qtzr.quant_info.usePerChannelMode
            and qtzr.tensor_quantizer_params
            and qtzr.quant_info.blockSize
        ):
            subcls = LPBQEncoding
        elif qtzr.data_type == QuantizationDataType.int:
            subcls = AffineEncoding
        else:
            subcls = FloatEncoding

        return subcls.from_quantizer(qtzr)


class AffineEncoding(EncodingBase):
    """
    Represents an affine quantization encoding.

                                              N..1
                                            ┌─────> 0.6.1 format
                    1..1                    | N..1
    QcQuantizeOp <--------> AffineEncoding ─┼─────> 1.0.0 format
                                            | 1..1
                                            └─────> 2.0.0 format


    Meaning of attributes are as follows:
      x_q = (x / scale - offset).round().clamp(qmin, qmax)
    """

    dtype: str
    channel_axis: int | Literal["auto"] | None = None
    block_axis: int | Literal["auto"] | None = None
    block_size: int | None = None

    def __init__(
        self,
        scale: np.ndarray,
        offset: np.ndarray,
        dtype: str,
        channel_axis: int | Literal["auto"] | None = None,
        block_axis: int | Literal["auto"] | None = None,
        block_size: int | None = None,
    ):
        self.scale = scale
        self.offset = offset
        self.dtype = dtype
        self.channel_axis = channel_axis
        self.block_axis = block_axis
        self.block_size = block_size

        if not isinstance(self.scale, np.ndarray):
            self.scale = np.array(self.scale, dtype=np.float64)

        if not isinstance(self.offset, np.ndarray):
            self.offset = np.array(self.offset, dtype=np.float64)

        if self.scale.shape != self.offset.shape:
            raise ValueError(
                f"Scale shape {self.scale.shape} does not match "
                f"offset shape {self.offset.shape}."
            )

        if self.channel_axis is not None and self.scale.ndim == 0:
            raise ValueError("Channel axis must be None for 0-dimensional scale")

        if self.channel_axis is None and self.scale.ndim != 0:
            if self.scale.shape == (1,):
                self.scale = self.scale.squeeze()
                self.offset = self.offset.squeeze()
            else:
                raise ValueError(
                    f"channel_axis must be specified for {self.scale.ndim}-dimensional scale"
                )

        if (self.block_axis is None) != (self.block_size is None):
            raise ValueError(
                "block_axis and block_size must be both specified or both None."
            )

        if isinstance(self.block_axis, int) and self.scale.ndim < 2:
            choices = " or ".join(
                ["None"] if self.scale.ndim == 0 else ["None", "'auto'"]
            )
            raise ValueError(
                f"block_axis must be {choices} for {self.scale.ndim}-dimensional scale."
            )

    def __repr__(self) -> str:
        attributes = [f"  dtype={self.dtype},"]

        if self.channel_axis is not None:
            attributes.append(f"  channel_axis={self.channel_axis},")

        if self.block_axis is not None:
            attributes.append(f"  block_axis={self.block_axis},")
            attributes.append(f"  block_size={self.block_size},")

        attributes += [
            f"  scale={self.scale},",
            f"  offset={self.offset},",
        ]
        return "\n".join(
            [
                "AffineEncoding(",
                *attributes,
                ")",
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineEncoding):
            return False
        return self.is_equal(other, allow_auto_axis=False)

    def is_equal(self, other: AffineEncoding, allow_auto_axis: bool = False) -> bool:
        return self._allclose(
            other, rtol=0.0, atol=0.0, allow_auto_axis=allow_auto_axis
        )

    def allclose(
        self,
        other: AffineEncoding,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        allow_auto_axis: bool = False,
    ) -> bool:
        return self._allclose(
            other, rtol=rtol, atol=atol, allow_auto_axis=allow_auto_axis
        )

    def _allclose(
        self,
        other: AffineEncoding,
        rtol: float,
        atol: float,
        allow_auto_axis: bool = False,
    ) -> bool:
        if (
            self.dtype != other.dtype
            or self.block_size != other.block_size
            or self.scale.size != other.scale.size
            or self.offset.size != other.offset.size
        ):
            return False

        channel_axis = other.channel_axis
        block_axis = other.block_axis
        scale = other.scale
        offset = other.offset

        if allow_auto_axis:
            if "auto" in (self.channel_axis, other.channel_axis):
                channel_axis = self.channel_axis

            if "auto" in (self.block_axis, other.block_axis):
                block_axis = self.block_axis

            scale = other.scale.reshape(self.scale.shape)
            offset = other.offset.reshape(self.offset.shape)

        return bool(
            self.channel_axis == channel_axis
            and self.block_axis == block_axis
            and np.allclose(self.scale, scale, rtol=rtol, atol=atol)
            and np.array_equal(self.offset, offset)
        )

    def to_signed(self) -> AffineEncoding:
        if self.signed:
            return self

        return AffineEncoding(
            scale=self.scale,
            offset=self.offset + 2 ** (self.bitwidth - 1),
            dtype=f"int{self.bitwidth}",
            channel_axis=self.channel_axis,
            block_axis=self.block_axis,
            block_size=self.block_size,
        )

    def to_unsigned(self) -> AffineEncoding:
        if not self.signed:
            return self

        return AffineEncoding(
            scale=self.scale,
            offset=self.offset - 2 ** (self.bitwidth - 1),
            dtype=f"uint{self.bitwidth}",
            channel_axis=self.channel_axis,
            block_axis=self.block_axis,
            block_size=self.block_size,
        )

    @property
    def signed(self) -> bool:
        unsigned, _ = self.dtype.split("int")
        return not bool(unsigned)

    @property
    def bitwidth(self) -> int:
        _, bitwidth = self.dtype.split("int")
        return int(bitwidth)

    @property
    def qmin(self) -> int:
        unsigned, bitwidth = self.dtype.split("int")

        if unsigned:
            return 0

        return -(2 ** (int(bitwidth) - 1))

    @property
    def qmax(self) -> int:
        unsigned, bitwidth = self.dtype.split("int")

        if unsigned:
            return 2 ** int(bitwidth) - 1

        return 2 ** (int(bitwidth) - 1) - 1

    @property
    def min(self) -> np.ndarray:
        """
        Returns the min value of the quantizer encoding
        """
        return (self.offset + self.qmin) * self.scale

    @property
    def max(self) -> np.ndarray:
        """
        Returns the min value of the quantizer encoding
        """
        return (self.offset + self.qmax) * self.scale

    def to_TfEncoding(self) -> list[libpymo.TfEncoding]:
        """
        Convert AffineEncoding object to list of TfEncoding objects.
        """
        tf_encodings = []
        bitwidth = self.bitwidth
        unsigned_encoding = self.to_unsigned()
        for scale, offset, min, max in zip(
            unsigned_encoding.scale.flatten(),
            unsigned_encoding.offset.flatten(),
            unsigned_encoding.min.flatten(),
            unsigned_encoding.max.flatten(),
        ):
            tf_encoding = libpymo.TfEncoding()
            tf_encoding.min = min
            tf_encoding.max = max
            tf_encoding.delta = scale
            tf_encoding.offset = offset
            tf_encoding.bw = bitwidth
            tf_encodings.append(tf_encoding)

        return tf_encodings

    def to_qnn_encoding_dict(
        self, encoding_version: str | None = None
    ) -> list | dict[str, Any]:
        if encoding_version == "0.6.1":
            return self._to_0_6_1()

        if encoding_version == "1.0.0":
            return self._to_1_0_0()

        if encoding_version == "2.0.0":
            return self._to_2_0_0()

        raise ValueError(
            f"Unsupported encoding version: {encoding_version}. "
            "Supported versions are: 0.6.1, 1.0.0, 2.0.0."
        )

    def _to_0_6_1(self) -> list[dict[str, Any]]:
        bitwidth = self.bitwidth

        return [
            {
                "min": min_,
                "max": max_,
                "scale": scale_,
                "offset": offset_,
                "bitwidth": bitwidth,
                "dtype": "int",
                "is_symmetric": str(self.signed),
            }
            for min_, max_, scale_, offset_ in zip(
                self.min.flatten().tolist(),
                self.max.flatten().tolist(),
                self.scale.flatten().tolist(),
                # 0.6.1 encoding offset assumes uint
                self.to_unsigned().offset.flatten().tolist(),
            )
        ]

    def _should_permute_to_1_0_0_blockwise_ordering(self) -> bool:
        """
        For Gemm and MatMul operators where the tensor is ordered (in_channels, out_channels),
        the 1.0.0 encoding format expects the encodings to be ordered in (out_channels, in_channels) order
        when blockwise quantization is used.

        This is a short-term fix to handle this until 1.0.0 format is deprecated.
        """
        # Note: This is a hacky way of preventing permute for ConvTranspose encodings,
        # since the quantizer is not aware of the operator type.
        return bool(
            self.block_size
            and self.scale.ndim == 2
            and isinstance(self.channel_axis, int)
            and self.channel_axis in (1, -1)
        )

    def _to_1_0_0(self) -> dict[str, Any]:
        scale = self.scale
        # 1.0.0 encoding offset assumes uint
        offset = self.to_unsigned().offset.astype(np.float64)

        if self._should_permute_to_1_0_0_blockwise_ordering():
            scale = scale.transpose((1, 0))
            offset = offset.transpose((1, 0))

        zero_point_shift = offset % 1.0
        offset = offset // 1.0

        encoding_dict = {
            "dtype": "INT",
            "bw": self.bitwidth,
            "is_sym": self.signed,
            "scale": scale.flatten().tolist(),
            "offset": offset.flatten().tolist(),
        }

        if np.any(zero_point_shift != 0.0):
            encoding_dict["zero_point_shift"] = zero_point_shift.flatten().tolist()

        if self.scale.ndim == 0:
            encoding_dict["enc_type"] = EncodingType.PER_TENSOR.name
        elif self.scale.ndim == 1 and not self.block_size:
            encoding_dict["enc_type"] = EncodingType.PER_CHANNEL.name
        else:
            encoding_dict["enc_type"] = EncodingType.PER_BLOCK.name
            encoding_dict["block_size"] = self.block_size

        return encoding_dict

    def _to_2_0_0(self) -> dict[str, Any]:
        y_scale = self.scale
        y_zero_point = -self.offset
        y_zero_point = y_zero_point.astype(
            np.int64 if np.all(y_zero_point % 1.0 == 0) else np.float64
        )

        if self.block_axis is not None:
            axis = self.block_axis
            block_size = self.block_size
        elif self.channel_axis is not None:
            axis = self.channel_axis
            block_size = None
            y_scale = y_scale.flatten()
            y_zero_point = y_zero_point.flatten()
        else:
            axis = None
            block_size = None
            y_scale = y_scale.squeeze()
            y_zero_point = y_zero_point.squeeze()

        if axis == "auto":
            raise RuntimeError(
                "AffineEncoding with axis='auto' cannot be "
                f"exported to 2.0.0 encoding format; got\n{self}"
            )

        y_scale = y_scale.tolist()
        y_zero_point = None if np.all(y_zero_point == 0) else y_zero_point.tolist()

        ret = {
            "output_dtype": self.dtype,
            "y_scale": y_scale,
        }
        if y_zero_point is not None:
            ret.update({"y_zero_point": y_zero_point})
        if axis is not None:
            ret.update({"axis": axis})
        if block_size is not None:
            ret.update({"block_size": block_size})

        return ret

    @classmethod
    def from_qnn_encoding_dict(
        cls,
        encoding_dict: list | dict[str, Any],
        input_shape: tuple[int, ...] | None = None,
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,
    ) -> AffineEncoding:
        version = cls._infer_encoding_version(encoding_dict)

        if version == "0.6.1":
            return cls._from_0_6_1(encoding_dict)
        if version == "1.0.0":
            return cls._from_1_0_0(
                encoding_dict,
                input_shape=input_shape,
                default_channel_axis=default_channel_axis,
                default_block_axis=default_block_axis,
            )
        else:
            return cls._from_2_0_0(
                encoding_dict,
                input_shape=input_shape,
                default_channel_axis=default_channel_axis,
                default_block_axis=default_block_axis,
            )

    @classmethod
    def _from_0_6_1(cls, encoding_dict) -> AffineEncoding:
        bitwidth = encoding_dict[0]["bitwidth"]
        signed = encoding_dict[0]["is_symmetric"] == "True"
        dtype = f"int{bitwidth}" if signed else f"uint{bitwidth}"

        scale = np.array(
            [enc["scale"] for enc in encoding_dict],
            dtype=np.float64,
        ).squeeze()
        offset = np.array(
            [enc["offset"] for enc in encoding_dict],
            dtype=np.float64,
        ).squeeze()

        if signed:
            offset += 2 ** (bitwidth - 1)

        channel_axis = None if scale.ndim == 0 else "auto"
        block_axis = None
        block_size = None

        return AffineEncoding(
            scale=scale,
            offset=offset,
            dtype=dtype,
            channel_axis=channel_axis,
            block_axis=block_axis,
            block_size=block_size,
        )

    @classmethod
    def _from_1_0_0(
        cls,
        encoding_dict,
        input_shape: tuple[int, ...] | None = None,
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,
    ) -> AffineEncoding:
        bitwidth = encoding_dict["bw"]
        signed = encoding_dict["is_sym"]
        dtype = f"int{bitwidth}" if signed else f"uint{bitwidth}"
        scale = np.array(encoding_dict["scale"], dtype=np.float64)
        offset = np.array(encoding_dict["offset"], dtype=np.float64)
        zero_point_shift = encoding_dict.get("zero_point_shift", 0.0)
        offset += zero_point_shift

        encoding_shape = scale.shape

        if encoding_dict["enc_type"] == EncodingType.PER_TENSOR.name:
            channel_axis = None
            block_axis = None
            block_size = None
            encoding_shape = []
        elif encoding_dict["enc_type"] == EncodingType.PER_CHANNEL.name:
            channel_axis = (
                "auto" if default_channel_axis is None else default_channel_axis
            )
            block_axis = None
            block_size = None
            encoding_shape = [scale.size]
        elif encoding_dict["enc_type"] == EncodingType.PER_BLOCK.name:
            channel_axis = (
                "auto" if default_channel_axis is None else default_channel_axis
            )
            block_axis = "auto" if default_block_axis is None else default_block_axis
            block_size = encoding_dict["block_size"]

            if (
                input_shape is not None
                and isinstance(channel_axis, int)
                and isinstance(block_axis, int)
            ):
                # Convert to positive index
                channel_axis = (len(input_shape) + channel_axis) % len(input_shape)
                block_axis = (len(input_shape) + block_axis) % len(input_shape)

                encoding_shape = [
                    dim
                    if axis == channel_axis
                    else dim // block_size
                    if axis == block_axis and block_size
                    else 1
                    for axis, dim in enumerate(input_shape)
                ]
        else:
            raise RuntimeError(f"Unsupported enc_type: {encoding_dict['enc_type']}")

        encoding = AffineEncoding(
            scale=scale.reshape(encoding_shape),
            offset=offset.reshape(encoding_shape),
            dtype=dtype,
            channel_axis=channel_axis,
            block_axis=block_axis,
            block_size=block_size,
        )

        # Legacy behavior is to shift offset by qmin
        encoding.offset -= encoding.qmin

        if encoding._should_permute_to_1_0_0_blockwise_ordering():
            # Reverse the temporary permutation done during export to 1.0.0 format
            encoding = AffineEncoding(
                scale=scale.reshape(*reversed(encoding_shape)).transpose((1, 0)),
                offset=offset.reshape(*reversed(encoding_shape)).transpose((1, 0)),
                dtype=dtype,
                channel_axis=channel_axis,
                block_axis=block_axis,
                block_size=block_size,
            )

        return encoding

    @classmethod
    def _from_2_0_0(
        cls,
        encoding_dict,
        input_shape: tuple[int, ...] | None = None,  # pylint: disable=unused-argument
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,  # pylint: disable=unused-argument
    ) -> AffineEncoding:
        if "per_block_int_scale" in encoding_dict:
            raise NotImplementedError("LPBQ encodings are not supported")

        scale = np.array(encoding_dict["y_scale"], dtype=np.float64)
        zp = encoding_dict.get("y_zero_point", None)

        if zp is None:
            offset = np.zeros_like(scale, dtype=np.float64)
        else:
            offset = -np.array(zp, dtype=np.float64)

        if "block_size" in encoding_dict:
            channel_axis = (
                "auto" if default_channel_axis is None else default_channel_axis
            )
            block_axis = encoding_dict["axis"]
            block_size = encoding_dict["block_size"]
        else:
            channel_axis = encoding_dict.get("axis", None)
            block_axis = None
            block_size = None

        return AffineEncoding(
            scale=scale,
            offset=offset,
            dtype=encoding_dict["output_dtype"],
            channel_axis=channel_axis,
            block_axis=block_axis,
            block_size=block_size,
        )

    @classmethod
    def from_quantizer(cls, qtzr: QcQuantizeOp) -> AffineEncoding | None:
        # pylint: disable=protected-access
        if not qtzr.enabled:
            return None

        if qtzr.quant_info.usePerChannelMode and qtzr.tensor_quantizer_params:
            channel_axis = qtzr.tensor_quantizer_params.channel_axis
            block_size = qtzr.quant_info.blockSize or None
            block_axis = (
                None if block_size is None else qtzr.tensor_quantizer_params.block_axis
            )
        else:
            channel_axis = None
            block_size = None
            block_axis = None

        encodings = qtzr.get_encodings()

        if encodings is None:
            # This means one of the three:
            #   1. This quantizer not enabled
            #   2. This quantizer not initialized
            #   3. This quantizer is a floating point quantizer
            # In any case, this corresponds to no-encoding in encoding_version 2.0.0
            return None

        signed = qtzr.use_symmetric_encodings
        bw = encodings[0].bw

        scale = np.array([e.delta for e in encodings])
        offset = np.array([e.offset for e in encodings])

        if signed:
            dtype = f"int{bw}"
            offset += 2 ** (bw - 1)
        else:
            dtype = f"uint{bw}"

        return AffineEncoding(
            scale=scale.reshape(qtzr._encoding_shape()),
            offset=offset.reshape(qtzr._encoding_shape()),
            dtype=dtype,
            channel_axis=channel_axis,
            block_axis=block_axis,
            block_size=block_size,
        )

    def load_to(self, qtzr: QcQuantizeOp) -> None:
        """
        Load encoding to QcQuantizeOp object
        """
        if (
            self.channel_axis is not None or self.block_axis is not None
        ) and qtzr.tensor_quantizer_params is None:
            raise RuntimeError(
                "QcQuantizeOp.tensor_quantizer_params is None; cannot set "
                "channel/block quantization."
            )

        if isinstance(self.channel_axis, int):
            qtzr.tensor_quantizer_params.channel_axis = self.channel_axis

        if isinstance(self.block_axis, int):
            qtzr.tensor_quantizer_params.block_axis = self.block_axis

        if self.channel_axis is None:
            qtzr.enable_per_channel_quantization(False)
        else:
            qtzr.enable_per_channel_quantization(True)

            if self.block_axis is None:
                # block_size=0 indicates no block quantization
                qtzr._enable_blockwise_quantization(block_size=0)  # pylint: disable=protected-access
            else:
                if not isinstance(self.block_size, int):
                    raise RuntimeError(
                        f"Cannot load encoding with block_size={self.block_size} to QcQuantizeOp."
                    )
                qtzr._enable_blockwise_quantization(block_size=self.block_size)  # pylint: disable=protected-access

        zero_point_shift = (self.offset % 1.0).flatten()
        if len(set(zero_point_shift.tolist())) > 1:
            raise RuntimeError(
                "Value of zero-point-shift must be the same for all encodings"
            )

        qtzr.bitwidth = self.bitwidth
        qtzr.use_symmetric_encodings = self.signed
        qtzr.use_strict_symmetric = self.signed and bool(np.all(self.offset == 1))
        qtzr.use_unsigned_symmetric = self.signed and bool(
            np.all(self.offset == 2 ** (self.bitwidth - 1))
        )
        qtzr.quant_info.tensorQuantizerRef.setEncodings(self.to_TfEncoding())
        qtzr.quant_info.tensorQuantizerRef.setZeroPointShift(zero_point_shift[0])
        qtzr.op_mode = libpymo.TensorQuantizerOpMode.quantizeDequantize


class LPBQEncoding(AffineEncoding):
    per_channel_float_scale: np.ndarray
    per_block_int_scale: np.ndarray
    dtype: str
    channel_axis: int | Literal["auto"]
    block_axis: int | Literal["auto"]
    block_size: int
    decompressed_dtype: str

    def decompressed_bitwidth(self) -> int:
        _, bitwidth = self.decompressed_dtype.split("int")
        return int(bitwidth)

    def __init__(  # pylint: disable=super-init-not-called
        self,
        per_channel_float_scale: np.ndarray,
        per_block_int_scale: np.ndarray,
        dtype: str,
        channel_axis: int | Literal["auto"] | None,
        block_axis: int | Literal["auto"] | None,
        block_size: int | None,
        decompressed_dtype: str | None = None,
    ):
        if per_block_int_scale.ndim == 2:
            if channel_axis == "auto" and isinstance(block_axis, int):
                channel_axis = (block_axis + 1) % 2

            if block_axis == "auto" and isinstance(channel_axis, int):
                block_axis = (channel_axis + 1) % 2

            if per_channel_float_scale.ndim == 1:
                per_channel_float_scale = per_channel_float_scale.reshape(
                    (per_channel_float_scale.size, 1)
                    if channel_axis in (0, -2)
                    else (1, per_channel_float_scale.size)
                )

        self.per_channel_float_scale = per_channel_float_scale
        self.per_block_int_scale = per_block_int_scale
        self.dtype = dtype
        self.channel_axis = channel_axis
        self.block_axis = block_axis
        self.block_size = block_size
        self.decompressed_dtype = decompressed_dtype or (
            f"int{self.bitwidth * 2}" if self.signed else f"uint{self.bitwidth * 2}"
        )

        if (
            self.channel_axis is None
            or self.block_axis is None
            or self.block_size is None
        ):
            raise ValueError(
                "LPBQEncoding requires channel_axis, block_axis and block_size to be specified; got "
                f"channel_axis={self.channel_axis}, "
                f"block_axis={self.block_axis}, "
                f"block_size={self.block_size}."
            )

        if self.per_block_int_scale.ndim != self.per_channel_float_scale.ndim:
            raise ValueError(
                "per_channel_float_scale and per_block_int_scale "
                "must have the same number of dimensions; got "
                f"per_channel_float_scale.ndim={self.per_channel_float_scale.ndim}, "
                f"per_block_int_scale.ndim={self.per_block_int_scale.ndim}, "
            )

        if isinstance(self.block_axis, int):
            if self.per_block_int_scale.ndim < 2:
                raise ValueError(
                    "block_axis must be 'auto' for "
                    f"{self.per_block_int_scale.ndim}-dimensional per_block_int_scale."
                )
            # Convert to positive index
            self.block_axis = (
                self.per_block_int_scale.ndim + self.block_axis
            ) % self.per_block_int_scale.ndim

            if not (
                self.per_channel_float_scale.shape[: self.block_axis]
                == self.per_block_int_scale.shape[: self.block_axis]
                and self.per_channel_float_scale.shape[self.block_axis + 1 :]
                == self.per_block_int_scale.shape[self.block_axis + 1 :]
            ):
                raise ValueError(
                    "per_channel_float_scale and per_block_int_scale shapes are incompatible; got "
                    f"per_channel_float_scale.shape={self.per_channel_float_scale.shape}, "
                    f"per_block_int_scale.shape={self.per_block_int_scale.shape}, "
                )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "LPBQEncoding(",
                f"  dtype={self.dtype},",
                f"  decompressed_dtype={self.decompressed_dtype},",
                f"  channel_axis={self.channel_axis},",
                f"  block_axis={self.block_axis},",
                f"  block_size={self.block_size},",
                f"  per_channel_float_scale={self.per_channel_float_scale},",
                f"  per_block_int_scale={self.per_block_int_scale},",
                ")",
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LPBQEncoding):
            return False
        return self.is_equal(other, allow_auto_axis=False)

    def is_equal(self, other: LPBQEncoding, allow_auto_axis: bool = False) -> bool:
        return self._allclose(
            other, rtol=0.0, atol=0.0, allow_auto_axis=allow_auto_axis
        )

    def allclose(
        self,
        other: LPBQEncoding,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        allow_auto_axis: bool = False,
    ) -> bool:
        return self._allclose(
            other, rtol=rtol, atol=atol, allow_auto_axis=allow_auto_axis
        )

    def _allclose(
        self,
        other: LPBQEncoding,
        rtol: float,
        atol: float,
        allow_auto_axis: bool = False,
    ) -> bool:
        if (
            self.dtype != other.dtype
            or self.block_size != other.block_size
            or self.per_block_int_scale.size != other.per_block_int_scale.size
            or self.per_channel_float_scale.size != other.per_channel_float_scale.size
        ):
            return False

        channel_axis = other.channel_axis
        block_axis = other.block_axis
        per_channel_float_scale = other.per_channel_float_scale
        per_block_int_scale = other.per_block_int_scale

        if allow_auto_axis:
            if "auto" in (self.channel_axis, other.channel_axis):
                channel_axis = self.channel_axis

            if "auto" in (self.block_axis, other.block_axis):
                block_axis = self.block_axis

            per_channel_float_scale = other.per_channel_float_scale.reshape(
                self.per_channel_float_scale.shape
            )
            per_block_int_scale = other.per_block_int_scale.reshape(
                self.per_block_int_scale.shape
            )

        return bool(
            self.channel_axis == channel_axis
            and self.block_axis == block_axis
            and np.allclose(
                self.per_channel_float_scale,
                per_channel_float_scale,
                rtol=rtol,
                atol=atol,
            )
            and np.array_equal(self.per_block_int_scale, per_block_int_scale)
        )

    def to_qnn_encoding_dict(
        self, encoding_version: str | None = None
    ) -> dict[str, Any]:
        if encoding_version == "1.0.0":
            return self._to_1_0_0()
        if encoding_version == "2.0.0":
            return self._to_2_0_0()

        raise ValueError(
            f"Unsupported encoding version: {encoding_version}. "
            "Supported versions are: 1.0.0, 2.0.0."
        )

    def _should_permute_to_1_0_0_blockwise_ordering(self) -> bool:
        """
        For Gemm and MatMul operators where the tensor is ordered (in_channels, out_channels),
        the 1.0.0 encoding format expects the encodings to be ordered in (out_channels, in_channels) order
        when blockwise quantization is used.

        This is a short-term fix to handle this until 1.0.0 format is deprecated.
        """
        # Note: This is a hacky way of preventing permute for ConvTranspose encodings,
        # since the quantizer is not aware of the operator type.
        return bool(
            self.block_size
            and self.per_block_int_scale.ndim == 2
            and isinstance(self.channel_axis, int)
            and self.channel_axis in (1, -1)
        )

    def _to_1_0_0(self) -> dict[str, Any]:
        per_block_int_scale = self.per_block_int_scale.astype(np.int64)

        if self._should_permute_to_1_0_0_blockwise_ordering():
            per_block_int_scale = per_block_int_scale.transpose((1, 0))

        offset = -(2 ** (self.decompressed_bitwidth() - 1))
        return {
            "dtype": "INT",
            "enc_type": EncodingType.LPBQ.name,
            "compressed_bw": self.bitwidth,
            "bw": self.decompressed_bitwidth(),
            "is_sym": True,
            "scale": self.per_channel_float_scale.flatten().tolist(),
            "per_block_int_scale": per_block_int_scale.flatten().tolist(),
            "offset": [offset] * self.per_channel_float_scale.size,
            "block_size": self.block_size,
        }

    def _to_2_0_0(self) -> dict[str, Any]:
        if self.decompressed_bitwidth() != self.bitwidth * 2:
            raise RuntimeError(
                "LPBQEncoding with decompressed_bitwidth != 2 * bitwidth cannot be "
                f"exported to 2.0.0 encoding format; got\n{self}"
            )

        return {
            "output_dtype": self.dtype,
            "per_channel_float_scale": self.per_channel_float_scale.tolist(),
            "per_block_int_scale": self.per_block_int_scale.astype(np.int64).tolist(),
            "axis": self.block_axis,
            "block_size": self.block_size,
        }

    @classmethod
    def from_qnn_encoding_dict(
        cls,
        encoding_dict: list | dict[str, Any],
        input_shape: tuple[int, ...] | None = None,
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,
    ) -> LPBQEncoding:
        version = cls._infer_encoding_version(encoding_dict)

        if version == "0.6.1":
            raise RuntimeError(
                "LPBQEncoding cannot be created from 0.6.1 encoding format."
            )
        elif version == "1.0.0":
            return cls._from_1_0_0(
                encoding_dict,
                input_shape=input_shape,
                default_channel_axis=default_channel_axis,
                default_block_axis=default_block_axis,
            )
        else:
            return cls._from_2_0_0(encoding_dict)

    @classmethod
    def _from_1_0_0(
        cls,
        encoding_dict,
        input_shape: tuple[int, ...] | None = None,
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,
    ) -> LPBQEncoding:
        bitwidth = encoding_dict["compressed_bw"]
        dtype = f"int{bitwidth}"
        decompressed_bw = encoding_dict["bw"]
        decompressed_dtype = f"int{decompressed_bw}"
        per_channel_float_scale = np.array(encoding_dict["scale"], dtype=np.float64)
        per_block_int_scale = np.array(
            encoding_dict["per_block_int_scale"], dtype=np.int32
        )
        block_size = encoding_dict["block_size"]
        channel_axis = "auto"
        block_axis = "auto"

        if (
            input_shape is not None
            and default_channel_axis is not None
            and default_block_axis is not None
        ):
            # Convert to positive index
            channel_axis = (len(input_shape) + default_channel_axis) % len(input_shape)
            block_axis = (len(input_shape) + default_block_axis) % len(input_shape)

            per_block_int_scale = per_block_int_scale.reshape(
                [
                    dim
                    if axis == channel_axis
                    else dim // block_size
                    if axis == block_axis
                    else 1
                    for axis, dim in enumerate(input_shape)
                ]
            )
            per_channel_float_scale = per_channel_float_scale.reshape(
                [
                    dim if axis == channel_axis else 1
                    for axis, dim in enumerate(input_shape)
                ]
            )

        encoding = LPBQEncoding(
            per_channel_float_scale=per_channel_float_scale,
            per_block_int_scale=per_block_int_scale,
            dtype=dtype,
            channel_axis=channel_axis,
            block_axis=block_axis,
            block_size=block_size,
            decompressed_dtype=decompressed_dtype,
        )

        if encoding._should_permute_to_1_0_0_blockwise_ordering():
            # Reverse the temporary permutation done during export to 1.0.0 format
            encoding = LPBQEncoding(
                per_channel_float_scale=per_channel_float_scale,
                per_block_int_scale=per_block_int_scale.reshape(
                    *reversed(encoding.per_block_int_scale.shape)
                ).transpose((1, 0)),
                dtype=dtype,
                channel_axis=channel_axis,
                block_axis=block_axis,
                block_size=block_size,
                decompressed_dtype=decompressed_dtype,
            )

        return encoding

    @classmethod
    def _from_2_0_0(
        cls,
        encoding_dict,
        input_shape: tuple[int, ...] | None = None,  # pylint: disable=unused-argument
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,  # pylint: disable=unused-argument
    ) -> LPBQEncoding:
        per_channel_float_scale = np.array(
            encoding_dict["per_channel_float_scale"], dtype=np.float64
        )
        per_block_int_scale = np.array(
            encoding_dict["per_block_int_scale"], dtype=np.int64
        )
        block_size = encoding_dict["block_size"]
        channel_axis = "auto" if default_channel_axis is None else default_channel_axis

        return LPBQEncoding(
            per_channel_float_scale=per_channel_float_scale,
            per_block_int_scale=per_block_int_scale,
            dtype=encoding_dict["output_dtype"],
            channel_axis=channel_axis,
            block_axis=encoding_dict["axis"],
            block_size=block_size,
        )

    @classmethod
    def from_quantizer(
        cls, qtzr: GroupedBlockQuantizeDequantize
    ) -> LPBQEncoding | None:
        # pylint: disable=protected-access
        if not qtzr.enabled:
            return None

        if qtzr.quant_info.usePerChannelMode and qtzr.tensor_quantizer_params:
            block_size = qtzr.quant_info.blockSize or None
        else:
            block_size = None

        if not block_size:
            raise RuntimeError("LPBQEncoding requires block_size to be specified.")

        channel_axis = qtzr.tensor_quantizer_params.channel_axis
        block_axis = qtzr.tensor_quantizer_params.block_axis

        encodings = qtzr.get_encodings()

        if encodings is None:
            # This means one of the three:
            #   1. This quantizer not enabled
            #   2. This quantizer not initialized
            #   3. This quantizer is a floating point quantizer
            # In any case, this corresponds to no-encoding in encoding_version 2.0.0
            return None

        scale, _ = lpbq_utils.encodings_to_scale_offset_arrays(
            encodings, qtzr._encoding_shape()
        )
        compressed_bw = qtzr.bitwidth
        decompressed_bw = qtzr.decompressed_bw
        per_block_int_scale, per_channel_scale = lpbq_utils.grouped_dynamic_quantize(
            scale, qtzr._block_grouping(), decompressed_bw - compressed_bw
        )

        per_channel_scale = per_channel_scale.squeeze(
            tuple(range(1, per_channel_scale.ndim, 2))
        )

        return LPBQEncoding(
            per_channel_float_scale=per_channel_scale,
            per_block_int_scale=per_block_int_scale,
            dtype=f"int{compressed_bw}",
            decompressed_dtype=f"int{decompressed_bw}",
            channel_axis=channel_axis,
            block_axis=block_axis,
            block_size=block_size,
        )

    def load_to(self, qtzr: QcQuantizeOp) -> None:
        """
        Load encoding to QcQuantizeOp object
        """
        from aimet_onnx.qc_quantize_op import GroupedBlockQuantizeDequantize

        if not isinstance(qtzr, GroupedBlockQuantizeDequantize):
            raise RuntimeError(
                "LPBQEncoding can only be loaded to GroupedBlockQuantizeDequantize quantizer."
            )

        if qtzr.tensor_quantizer_params is None:
            raise RuntimeError(
                "QcQuantizeOp.tensor_quantizer_params is None; cannot set "
                "channel/block quantization."
            )

        qtzr.decompressed_bw = self.decompressed_bitwidth()
        return super().load_to(qtzr)

    @property
    def scale(self) -> np.ndarray:
        """
        Get the effective scale of the LPBQEncoding.
        """
        try:
            return self.per_block_int_scale * self.per_channel_float_scale
        except ValueError as e:
            raise ValueError(
                "Failed to compute combined scale from per_channel_float_scale and "
                "per_block_int_scale due to incompatible shapes; got "
                f"per_channel_float_scale.shape={self.per_channel_float_scale.shape}, "
                f"per_block_int_scale.shape={self.per_block_int_scale.shape}. "
            ) from e

    @property
    def offset(self) -> np.ndarray:
        centroid = math.ceil((self.qmax + self.qmin) / 2)
        return np.full(self.per_block_int_scale.shape, -centroid, dtype=np.float64)

    def to_signed(self) -> AffineEncoding:
        if self.signed:
            return self

        return LPBQEncoding(
            per_channel_float_scale=self.per_channel_float_scale,
            per_block_int_scale=self.per_block_int_scale,
            dtype=f"int{self.bitwidth}",
            channel_axis=self.channel_axis,
            block_axis=self.block_axis,
            block_size=self.block_size,
            decompressed_dtype=f"int{self.decompressed_bitwidth()}",
        )

    def to_unsigned(self) -> LPBQEncoding:
        if not self.signed:
            return self

        return LPBQEncoding(
            per_channel_float_scale=self.per_channel_float_scale,
            per_block_int_scale=self.per_block_int_scale,
            dtype=f"uint{self.bitwidth}",
            channel_axis=self.channel_axis,
            block_axis=self.block_axis,
            block_size=self.block_size,
            decompressed_dtype=f"uint{self.decompressed_bitwidth()}",
        )


@dataclass(frozen=True)
class FloatEncoding(EncodingBase):
    exponent_bits: int
    mantissa_bits: int
    finite: bool = False
    unsigned_zero: bool = False

    def to_qnn_encoding_dict(
        self, encoding_version: str | None = None
    ) -> list | dict[str, Any]:
        if encoding_version == "0.6.1":
            return self._to_0_6_1()

        if encoding_version == "1.0.0":
            return self._to_1_0_0()

        if encoding_version == "2.0.0":
            return self._to_2_0_0()

        raise ValueError(
            f"Unsupported encoding version: {encoding_version}. "
            "Supported versions are: 0.6.1, 1.0.0"
        )

    def _to_0_6_1(self) -> list[dict[str, Any]]:
        if self == _float16:
            return [{"bitwidth": 16, "dtype": "float"}]
        raise RuntimeError

    def _to_1_0_0(self) -> dict[str, Any]:
        if self == _float16:
            return {
                "dtype": "FLOAT",
                "bw": 16,
                "enc_type": EncodingType.PER_TENSOR.name,
            }
        raise RuntimeError

    def _to_2_0_0(self) -> dict[str, Any]:
        if self in (_float16, _bfloat16):
            return {}

        raise RuntimeError("FloatEncoding cannot be exported to 2.0.0 encoding format.")

    @classmethod
    def from_qnn_encoding_dict(
        cls,
        encoding_dict: list | dict[str, Any],
        input_shape: tuple[int, ...] | None = None,
        default_channel_axis: int | None = None,
        default_block_axis: int | None = None,
    ) -> FloatEncoding:
        version = cls._infer_encoding_version(encoding_dict)

        if version == "0.6.1":
            return cls._from_0_6_1(encoding_dict)
        else:
            return cls._from_1_0_0(encoding_dict)

    @classmethod
    def _from_0_6_1(cls, encoding_dict) -> FloatEncoding:
        if encoding_dict == _float16.to_qnn_encoding_dict("0.6.1"):
            return _float16
        raise RuntimeError

    @classmethod
    def _from_1_0_0(cls, encoding_dict) -> FloatEncoding:
        encoding_dict = encoding_dict.copy()
        encoding_dict.pop("name", None)
        if encoding_dict == _float16.to_qnn_encoding_dict("1.0.0"):
            return _float16
        raise RuntimeError

    @classmethod
    def from_quantizer(cls, qtzr: QcQuantizeOp) -> FloatEncoding | None:
        if qtzr.data_type != QuantizationDataType.float:
            raise RuntimeError(
                f"Can't create FloatEncoding from QcQuantizeOp with data_type={qtzr.data_type}"
            )

        if not qtzr.enabled:
            return None

        if qtzr.bitwidth == 16:
            return _float16

        raise NotImplementedError(
            f"FloatEncoding.from_quantizer only supports float16; got bitwidth={qtzr.bitwidth}."
        )

    def load_to(self, qtzr: QcQuantizeOp) -> None:
        """
        Load encoding to QcQuantizeOp object
        """
        if self == _float16:
            qtzr.data_type = QuantizationDataType.float
            qtzr.bitwidth = 16
            qtzr.enabled = True
            return

        raise NotImplementedError(
            f"FloatEncoding.load_to only supports float16; got\n{self}"
        )


# ONNX floating point data types
_float16 = FloatEncoding(
    exponent_bits=5, mantissa_bits=10, finite=False, unsigned_zero=False
)
_bfloat16 = FloatEncoding(
    exponent_bits=8, mantissa_bits=7, finite=False, unsigned_zero=False
)
_float8e4m3fn = FloatEncoding(
    exponent_bits=4, mantissa_bits=3, finite=True, unsigned_zero=False
)
_float8e4m3fnuz = FloatEncoding(
    exponent_bits=4, mantissa_bits=3, finite=True, unsigned_zero=True
)
_float8e5m2 = FloatEncoding(
    exponent_bits=5, mantissa_bits=2, finite=False, unsigned_zero=False
)
_float8e5m2fnuz = FloatEncoding(
    exponent_bits=5, mantissa_bits=2, finite=True, unsigned_zero=True
)
