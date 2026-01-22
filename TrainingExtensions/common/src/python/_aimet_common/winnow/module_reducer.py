# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Module reducer abstract class."""

import abc
from typing import Dict, List

from ..connected_graph.operation import Op
from ..winnow.mask import Mask


class ModuleReducer(abc.ABC):
    """The ModuleReducer class contains functionality to reduce a module's weight parameter and adjust the module's
    number of input and output channels.
    """

    def __init__(
        self, using_cuda: bool, reshape: bool, op_to_mask_dict: Dict[Op, Mask]
    ):
        """
        ModuleReducer initialization.

        :param using_cuda: Indicates if a module is on GPU.
        :param reshape: If True, ModuleReducer will add DownsampleLayer and UpsampleLayer as needed.
                        If False, ModuleReducer will not add DownsampleLayer and UpsampleLayer.
        :param op_to_mask_dict: Dictionary mapping Op to mask
        """

        self._using_cuda = using_cuda
        self._reshape = reshape
        self._op_to_mask_dict = op_to_mask_dict

    @abc.abstractmethod
    def reduce_modules(self, list_of_ops_to_reduce: List):
        """
        For the Ops in the list, reduce the corresponding module.

        :param list_of_ops_to_reduce: list of Ops whose associated modules need to be reduced.
        """
