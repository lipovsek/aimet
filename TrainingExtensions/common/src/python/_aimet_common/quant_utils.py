# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities for quantization"""

from typing import Dict
import numpy as np

from .defs import QuantizationDataType, EncodingType
from .utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def get_conv_accum_bounds(weights: np.ndarray, quant_bw: int, accum_bw: int):
    """
    Get upper and lower bounds for accumulator for a given layer
    :param weights: Weight tensor in OIHW format
    :param quant_bw: Quantization bitwidth
    :param accum_bw: Accumulator bitwidth
    :return: Tuple of (was accumulator range exceeded, most accumulator range used)
    """

    # Max integer value
    max_int_value = (2**quant_bw) - 1
    max_accum_value = 2 ** (accum_bw - 1)

    # Calculate min and max (absolute)
    quant_min = min(np.min(weights), 0)
    quant_max = max(np.max(weights), 0)
    quant_scale = 2 * max(abs(quant_min), abs(quant_max)) / max_int_value
    if quant_scale == 0:
        quant_scale = 1e-5  # Prevent divide by zero for degenerate layers

    most_accum_range_used = 0
    was_accum_range_exceeded = False

    for out_chan_index in range(weights.shape[0]):
        accum_max = np.sum(
            max_int_value
            * np.maximum(np.round(weights[out_chan_index] / quant_scale), 0)
        )
        accum_min = np.sum(
            max_int_value
            * np.minimum(np.round(weights[out_chan_index] / quant_scale), 0)
        )

        if accum_max / max_accum_value > most_accum_range_used:
            most_accum_range_used = accum_max / max_accum_value

        if accum_min / -max_accum_value > most_accum_range_used:
            most_accum_range_used = accum_min / -max_accum_value

        if (accum_max >= max_accum_value) or (accum_min < -max_accum_value):
            was_accum_range_exceeded = True
            _logger.info(
                "Accumulator range potentially exceeded in channel %d", out_chan_index
            )

    return was_accum_range_exceeded, most_accum_range_used


def _convert_encoding_format_0_6_1_to_1_0_0(encodings: Dict[str, Dict]):
    """
    Convert encoding dictionary in 0.6.1 format to 1.0.0 format (list of encodings)
    """
    new_encodings = []
    assert isinstance(encodings, dict)
    for tensor_name, encoding in encodings.items():
        new_encoding = {"name": tensor_name}
        new_encoding["bw"] = encoding[0]["bitwidth"]
        if encoding[0]["dtype"] == "float":
            new_encoding["dtype"] = QuantizationDataType.float.name.upper()
            new_encoding["enc_type"] = EncodingType.PER_TENSOR.name
        else:
            new_encoding["dtype"] = QuantizationDataType.int.name.upper()
            new_encoding["is_sym"] = encoding[0]["is_symmetric"] == "True"
            if len(encoding) == 1:
                new_encoding["enc_type"] = EncodingType.PER_TENSOR.name
            else:
                new_encoding["enc_type"] = EncodingType.PER_CHANNEL.name

            scale = []
            offset = []
            for enc in encoding:
                scale.append(enc["scale"])
                offset.append(enc["offset"])
            new_encoding["scale"] = scale
            new_encoding["offset"] = offset
        new_encodings.append(new_encoding)

    return new_encodings
