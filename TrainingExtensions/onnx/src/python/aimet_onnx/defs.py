# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Definitions for ONNX"""

import numpy as np


class DataLoader:
    """
    Example of a Dataloader which can be used for running AMPv2 and AutoQuantv2
    """

    def __init__(self, data: np.ndarray, batch_size: int, iterations: int):
        """
        :param data: Numpy array
        :param batch_size: batch size for data loader
        :param iterations: number of iterations
        """
        self._data = data
        self.batch_size = batch_size
        self.iterations = iterations

    def __iter__(self):
        """Iterates over dataset"""

    def __len__(self):
        """Returns number of batches the dataloader will iterate"""
        return self.iterations
