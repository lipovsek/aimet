# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin
"""Base encoding definition"""

import abc
import copy

import torch


__all__ = ["EncodingBase"]


class EncodingBase(abc.ABC):
    """
    Quantizer encoding base class
    """

    @property
    @abc.abstractmethod
    def scale(self) -> torch.Tensor:
        """
        Returns quantization scale
        """

    @property
    @abc.abstractmethod
    def block_size(self) -> tuple[int, ...] | None:
        """
        Returns quantization block size
        """

    @property
    @abc.abstractmethod
    def bitwidth(self) -> int:
        """
        Returns the bitwidth of the quantized representation
        """

    @property
    @abc.abstractmethod
    def mapping(self) -> str:
        """
        Returns the type of mapping function of this encoding object
        """

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        """
        Changes dtype of data in quantizer encoding or device where the data is.
        Returns new encoding with changed dtype and device without changing current encoding
        as `torch.Tensor.to`.
        """

    @abc.abstractmethod
    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input with the encoding

        :param input: Tensor to be quantized
        :return: Quantized tensor
        """

    @abc.abstractmethod
    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Dequantize the input with the encoding

        :param input: Tensor to be dequantized
        :return: Dequantized tensor
        """

    @abc.abstractmethod
    def quantize_dequantize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Quantize-dequantize the input with the encoding

        :param input: Tensor to be dequantized
        :return: Quantize-dequantized tensor
        """

    def _detach(self) -> "EncodingBase":
        """
        Returns a new encoding object with all tensors attributes detached from the current graph
        """
        self_copy = copy.copy(self)
        for name, item in self_copy.__dict__.items():
            if isinstance(item, torch.Tensor):
                setattr(self_copy, name, item.detach())
        return self_copy

    def _clone(self) -> "EncodingBase":
        """
        Returns a new encoding object all tensor attributes cloned
        """
        self_copy = copy.copy(self)
        for name, item in self_copy.__dict__.items():
            if isinstance(item, torch.Tensor):
                setattr(self_copy, name, item.clone())
        return self_copy

    def to_qnn_encoding_dict(self, encoding_version=None):
        """
        Converts encoding object into QNN encoding dictionary
        """
        raise NotImplementedError

    @classmethod
    def _from_qnn_encoding_dict(cls, encoding_dict, version=None) -> "EncodingBase":
        """
        Create an encoding object from a QNN encoding dictionary
        """
        raise NotImplementedError

    @property
    def granularity(self) -> str:
        """
        Returns the granularity of the quantizer encoding
        """
        if self.scale.dim() == 0:
            return "pertensor"
        if self.block_size is not None:
            return "blockwise"
        non_singleton_dims = tuple(dim for dim in self.scale.shape if dim > 1)
        if len(non_singleton_dims) <= 1:
            return "perchannel"
        return "unknown"

    def _get_channel_axis(self) -> int | None:
        try:
            channel_axis = next(
                iter(axis for axis, dim in enumerate(self.scale.shape) if dim > 1)
            )
        except StopIteration:
            # Per-channel encoding that happens to have only one output channel
            # In this case, fall back to per-tensor encoding since we aren't fully
            # sure about the channel axis
            channel_axis = None
        return channel_axis

    def _get_block_axis(self) -> int | None:
        # NOTE: DO NOT USE THIS FUNCTION except for QNN encoding export.
        #       This function assumes block axis can only be either axis 0 or axis 1.
        #       This assumption holds in practical cases, but does not cover all theoretically
        #       possible cases.
        if self.block_size is None:
            raise RuntimeError

        for axis, blk in enumerate(self.block_size[:2]):
            if blk != 1:
                return axis

        return None
