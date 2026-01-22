# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Type definitions that are used across GPTVQ"""

from dataclasses import dataclass
from typing import Callable, Any, Optional

from torch import nn
from torch.utils.data import DataLoader

GPTVQSupportedModules = (nn.Linear, nn.Conv2d)
DAMPENING_PERCENTAGE = 0.01
BLOCK_STRIDE = 128


@dataclass
class GPTVQParameters:
    """
    Data carrier containing GPTVQ parameters
    """

    # pylint: disable=too-many-instance-attributes
    data_loader: DataLoader
    forward_fn: Callable[[nn.Module, Any], Any]
    row_axis: int = 0
    col_axis: int = 1
    rows_per_block: int = 32
    cols_per_block: int = 256
    vector_dim: int = 2
    vector_bw: int = 8
    vector_stride: int = 1
    index_bw: int = 6
    num_of_kmeans_iterations: int = 100
    assignment_chunk_size: Optional[int] = None
