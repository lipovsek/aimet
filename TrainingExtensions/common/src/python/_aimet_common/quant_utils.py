# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities for quantization"""

import numpy as np

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
