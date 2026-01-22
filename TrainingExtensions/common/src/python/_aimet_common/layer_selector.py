# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Selects layers for compression based on different criteria"""

import abc
from typing import List

from .layer_database import LayerDatabase


class LayerSelector(abc.ABC):
    """
    Enables layers to be selected for compression
    """

    def select(self, layer_db: LayerDatabase, modules_to_ignore: List):
        """
        Marks certain layers as selected in a given layerDatabase according to a selection criteria
        :param layer_db: Layer database
        :param modules_to_ignore: Modules to skip for selection
        :return: No return
        """
