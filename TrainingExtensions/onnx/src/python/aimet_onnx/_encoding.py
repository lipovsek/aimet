# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=redefined-builtin
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any, Literal, TypeVar, Type
from aimet_onnx.common.defs import EncodingType
from aimet_onnx.common import libpymo


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
    ) -> T:
        """
        Create EncodingBase object from QNN encoding dict format.

        Args:
            encoding_dict: QNN encoding dict
        """
        version = cls._infer_encoding_version(encoding_dict)

        if version == "0.6.1":
            subcls = (
                AffineEncoding if encoding_dict[0]["dtype"] == "int" else FloatEncoding
            )
        elif version == "1.0.0":
            subcls = (
                AffineEncoding if encoding_dict["dtype"] == "INT" else FloatEncoding
            )
        else:
            subcls = (
                AffineEncoding
                if "int" in encoding_dict["output_dtype"]
                else FloatEncoding
            )

        return subcls.from_qnn_encoding_dict(encoding_dict)

    @abstractmethod
    def to_TfEncoding(self) -> list[libpymo.TfEncoding]:
        """
        Convert EncodingBase object to list of TfEncoding objects.
        """

    @classmethod
    def _infer_encoding_version(cls, encoding_dict: list | dict[str, Any]):
        if isinstance(encoding_dict, list):
            version = "0.6.1"
        else:
            version = "1.0.0" if "bw" in encoding_dict else "2.0.0"

        return version


@dataclass
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

    scale: np.ndarray
    offset: np.ndarray
    dtype: str
    channel_axis: int | Literal["auto"] | None = None
    block_axis: int | Literal["auto"] | None = None
    block_size: int | None = None

    def __post_init__(self):
        if not isinstance(self.scale, np.ndarray):
            self.scale = np.array(self.scale, dtype=np.float32)

        if not isinstance(self.offset, np.ndarray):
            self.offset = np.array(self.offset, dtype=np.float32)

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
                    f"Channel axis must be specified for {self.scale.ndim}-dimensional scale"
                )

        if (self.block_axis is None) != (self.block_size is None):
            raise ValueError(
                "block_axis and block_size must be both specified or both None."
            )

        if isinstance(self.block_axis, int) and self.scale.ndim < 2:
            if self.scale.ndim == 0:
                raise ValueError("Block axis must be None for 0-dimensional scale.")
            else:
                raise ValueError(
                    f"Block axis must be 'auto' or None for {self.scale.ndim}-dimensional scale."
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

        return (
            self.channel_axis == channel_axis
            and self.block_axis == block_axis
            and np.all(self.scale == scale)
            and np.all(self.offset == offset)
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
        symmetric = self.signed and np.all(self.offset == 0)

        return [
            {
                "min": min_,
                "max": max_,
                "scale": scale_,
                "offset": offset_,
                "bitwidth": bitwidth,
                "dtype": "int",
                "is_symmetric": str(symmetric),
            }
            for min_, max_, scale_, offset_ in zip(
                self.min.flatten().tolist(),
                self.max.flatten().tolist(),
                self.scale.flatten().tolist(),
                # 0.6.1 encoding offset assumes uint
                self.to_unsigned().offset.flatten().tolist(),
            )
        ]

    def _to_1_0_0(self) -> dict[str, Any]:
        # 1.0.0 encoding offset assumes uint
        offset = self.to_unsigned().offset.astype(np.float64)
        zero_point_shift = offset % 1.0
        offset = offset // 1.0

        encoding_dict = {
            "dtype": "INT",
            "bw": self.bitwidth,
            "is_sym": self.signed and np.all(self.offset == 0),
            "scale": self.scale.flatten().tolist(),
            "offset": offset.flatten().tolist(),
        }

        if np.any(zero_point_shift != 0.0):
            encoding_dict["zero_point_shift"] = zero_point_shift.flatten().tolist()

        if self.scale.ndim == 0:
            encoding_dict["enc_type"] = EncodingType.PER_TENSOR.name
        elif self.scale.ndim == 1:
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
        cls, encoding_dict: list | dict[str, Any]
    ) -> AffineEncoding:
        version = cls._infer_encoding_version(encoding_dict)

        if version == "0.6.1":
            return cls._from_0_6_1(encoding_dict)
        if version == "1.0.0":
            return cls._from_1_0_0(encoding_dict)
        else:
            return cls._from_2_0_0(encoding_dict)

    @classmethod
    def _from_0_6_1(cls, encoding_dict) -> AffineEncoding:
        bitwidth = encoding_dict[0]["bitwidth"]
        signed = encoding_dict[0]["is_symmetric"] == "True"
        dtype = f"int{bitwidth}" if signed else f"uint{bitwidth}"

        scale = np.array(
            [enc["scale"] for enc in encoding_dict],
            dtype=np.float32,
        ).squeeze()
        offset = np.array(
            [enc["offset"] for enc in encoding_dict],
            dtype=np.float32,
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
    def _from_1_0_0(cls, encoding_dict) -> AffineEncoding:
        bitwidth = encoding_dict["bw"]
        signed = encoding_dict["is_sym"]
        dtype = f"int{bitwidth}" if signed else f"uint{bitwidth}"
        scale = np.array(encoding_dict["scale"], dtype=np.float32)
        offset = np.array(encoding_dict["offset"], dtype=np.float64)
        zero_point_shift = encoding_dict.get("zero_point_shift", 0.0)
        offset += zero_point_shift

        if encoding_dict["enc_type"] == EncodingType.PER_TENSOR.name:
            channel_axis = None
            block_axis = None
            block_size = None
        elif encoding_dict["enc_type"] == EncodingType.PER_CHANNEL.name:
            channel_axis = "auto"
            block_axis = None
            block_size = None
        elif encoding_dict["enc_type"] == EncodingType.PER_BLOCK.name:
            channel_axis = "auto"
            block_axis = "auto"
            block_size = encoding_dict["block_size"]
        else:
            raise RuntimeError(f"Unsupported enc_type: {encoding_dict['enc_type']}")

        encoding = AffineEncoding(
            scale=scale,
            offset=offset,
            dtype=dtype,
            channel_axis=channel_axis,
            block_axis=block_axis,
            block_size=block_size,
        )

        # Legacy behavior is to shift offset by qmin
        encoding.offset -= encoding.qmin
        return encoding

    @classmethod
    def _from_2_0_0(cls, encoding_dict) -> AffineEncoding:
        if "per_block_int_scale" in encoding_dict:
            raise NotImplementedError("LPBQ encodings are not supported")

        scale = np.array(encoding_dict["y_scale"], dtype=np.float32)
        zp = encoding_dict.get("y_zero_point", None)

        if zp is None:
            offset = np.zeros_like(scale, dtype=np.float32)
        else:
            offset = -np.array(zp, dtype=np.float32)

        if "block_size" in encoding_dict:
            channel_axis = "auto"
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


class FloatEncoding(EncodingBase): ...
