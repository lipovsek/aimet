# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Mask propagation winnower abstract class"""

import logging
from abc import ABC, abstractmethod
from ..utils import AimetLogger


class MaskPropagationWinnower(ABC):
    """Abstract MaskPropagationWinnower class"""

    def __init__(self, list_of_modules_to_winnow, reshape, in_place, verbose):
        self._list_of_modules_to_winnow = list_of_modules_to_winnow
        self._reshape = reshape
        self._in_place = in_place

        if verbose is True:
            AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Winnow, logging.INFO)
        else:
            AimetLogger.set_area_logger_level(
                AimetLogger.LogAreas.Winnow, logging.WARNING
            )

    @abstractmethod
    def propagate_masks_and_winnow(self):
        """Propagate masks through the connected graph and perform winnowing"""
