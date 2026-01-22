# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Channel Pruning functions"""

import numpy as np


def select_channels_to_prune(
    weight_data: np.array, comp_ratio: float, num_in_channels: int
) -> list:
    """
    Based on the weight date, compression ratio and the number of input channels, return the
    input channel indices to prune.

    :param weight_data: numpy array. weight data to use to select the channels to be pruned.
    :param comp_ratio: compression ratio to compress
    :param num_in_channels: the number of input channels for the module
    :return: list of input channel indices that must be pruned.
    """

    assert 0 < comp_ratio <= 1

    # Weight data is of shape [Noc, Nic, k_h, k_w]

    # Calculate squared magnitudes of weight data along all the axis except input channels (dim = 1)
    magnitudes = np.sum(np.sum(np.sum((weight_data**2), 3), 2), 0)

    all_indices = list(range(num_in_channels))

    # get number of input channels to keep
    keep_inp_channels = max(1, int(num_in_channels * comp_ratio))

    # get the top indices
    keep_indices = np.argpartition(magnitudes, -keep_inp_channels)[-keep_inp_channels:]

    # get input channel indices to prune
    prune_indices = list(set(all_indices) - set(keep_indices))
    return prune_indices
