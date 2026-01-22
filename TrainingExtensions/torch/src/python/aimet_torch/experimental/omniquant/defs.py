# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Defs for dataclass"""

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class _LetPair:
    """
    A pair of modules for Omniquant LET optimization.
    prev: previous module in LET pair.
    follow: following module in LET pair.
    """

    prev: List[torch.nn.Module]
    follow: List[torch.nn.Module]

    def __str__(self):
        # Print LetPair info
        return f"LetPair(\n    prev: '{self.prev}',\n    follow: '{self.follow}',\n)"
