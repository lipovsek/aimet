# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utility functions for applying LPBQ quantization"""

from typing import List, Tuple, Sequence
import numpy as np

from aimet_onnx import qtype
from aimet_onnx.common import libpymo
from aimet_onnx.utils import numpy_to_TfEncoding, numpy_from_TfEncoding


def _split_blocks(encoding: np.ndarray, block_grouping) -> np.ndarray:
    """
    Get expanded scale shape which breaks each scale dimension into a pair of dimensions with sizes
    (original_shape / block_grouping, block_grouping).

    :return: Expanded scale shape
    """
    expanded_shape = []
    for idx, block_group in enumerate(block_grouping):
        # Block group of -1 is equivalent to grouping all blocks together
        if block_group == -1:
            expanded_shape.append(1)
            expanded_shape.append(encoding.shape[idx])
        else:
            expanded_shape.append(encoding.shape[idx] // block_group)
            expanded_shape.append(block_group)
    return encoding.reshape(expanded_shape)


def _get_per_group_scale_factor(
    scale: np.ndarray, block_grouping: Sequence[int], scale_bitwidth: int
) -> np.ndarray:
    """
    Get per channel scale.

    :param scale: Scale array
    :param block_grouping: Number of indices to group together for each dimension of scale
    :return: Per-group scale factor
    """
    grouped_scale = _split_blocks(scale, block_grouping)
    group_axes = tuple(range(1, len(grouped_scale.shape), 2))
    max_scale = np.max(grouped_scale, axis=group_axes, keepdims=True)
    per_group_scale = max_scale / 2**scale_bitwidth
    return per_group_scale


def grouped_dynamic_quantize(
    input_array: np.ndarray, grouping, bitwidth
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize input array between (1, 2 ** bitwidth) based on the maximum value in each group.

    :input_array: numpy array to quantize
    :param grouping: Number of indices per group for each axis of the input_array
    :bitwidth: Quantization bitwidth
    :return: Tuple of quantized input and quantization scale
    """
    dynamic_scale = _get_per_group_scale_factor(input_array, grouping, bitwidth)
    grouped_scale = _split_blocks(input_array, grouping)
    # Note: following aimet_torch implementation, clip to 2 ** bitwidth
    quantized_input = np.clip(
        np.round(grouped_scale / dynamic_scale), 1, 2**bitwidth
    ).astype(np.uint32)
    return quantized_input.reshape(input_array.shape), dynamic_scale


def compress_encoding_scales(
    encodings: List[libpymo.TfEncoding],
    encoding_shape: Sequence[int],
    block_grouping: Sequence[int],
    scale_bitwidth: int,
) -> List[libpymo.TfEncoding]:
    """
    Performs dynamic quantize-dequantization on encodings with the granularity specified in block_grouping

    :param encodings: Encodings to quantize-dequantize
    :param encoding_shape: Shape of encodings
    :param block_grouping: Number of indices at each axis of the encoding_shape to be grouped together
    :param scale_bitwidth: Bitwidth of quantize-dequantize operation to be performed on the encoding scales
    """
    assert len(encoding_shape) == len(block_grouping)
    scale, offset = numpy_from_TfEncoding(encodings, encoding_shape)
    compressed_scales = _compress_encoding_scales(scale, block_grouping, scale_bitwidth)
    new_encodings = numpy_to_TfEncoding(
        compressed_scales, offset, qtype.int(encodings[0].bw)
    )
    return new_encodings


def _compress_encoding_scales(
    scale: np.ndarray, block_grouping: Sequence[int], scale_bitwidth: int
) -> np.ndarray:
    int_scale, per_group_scale_factor = grouped_dynamic_quantize(
        scale, block_grouping, scale_bitwidth
    )
    grouped_int_scale = _split_blocks(int_scale, block_grouping)
    dequantized_scale = grouped_int_scale * per_group_scale_factor
    return dequantized_scale.reshape(scale.shape)
