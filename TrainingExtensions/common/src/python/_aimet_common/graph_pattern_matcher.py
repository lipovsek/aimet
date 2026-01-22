# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Main class for pattern matcher"""

from .utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


class PatternType:
    """
    structure to hold pattern data type
    """

    def __init__(self, pattern, action):
        """
        PatternType class holds a pattern with a corresponding actions
        :param pattern: pattern to be searched
        :param action: action to be applied upon finding pattern
        """
        self.pattern = pattern
        self.action = action
