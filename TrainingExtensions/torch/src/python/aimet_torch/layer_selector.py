# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Selects layers for compression based on different criteria"""

from typing import List

import torch
import torch.nn

from aimet_torch.common.defs import LayerCompRatioPair
from aimet_torch.common.layer_database import LayerDatabase
from aimet_torch.common.layer_selector import LayerSelector


class ConvFcLayerSelector(LayerSelector):
    """
    Selects conv and fc layers
    """

    def select(self, layer_db: LayerDatabase, modules_to_ignore: List[torch.nn.Module]):
        selected_layers = []
        for layer in layer_db:
            if layer.module in modules_to_ignore:
                continue

            if isinstance(layer.module, torch.nn.Linear):
                selected_layers.append(layer)

            elif isinstance(layer.module, torch.nn.Conv2d) and (
                layer.module.groups == 1
            ):
                selected_layers.append(layer)

        layer_db.mark_picked_layers(selected_layers)


class ConvNoDepthwiseLayerSelector(LayerSelector):
    """
    Selects conv layers (non-depthwise) for compression
    """

    def select(self, layer_db: LayerDatabase, modules_to_ignore: List[torch.nn.Module]):
        selected_layers = []
        for layer in layer_db:
            if layer.module in modules_to_ignore:
                continue

            if isinstance(layer.module, torch.nn.Conv2d) and (layer.module.groups == 1):
                selected_layers.append(layer)

        layer_db.mark_picked_layers(selected_layers)


class ManualLayerSelector(LayerSelector):
    """
    Marks layers which were manually selected and passed in by the user
    """

    def __init__(self, layer_comp_ratio_pairs: List[LayerCompRatioPair]):
        self._layer_comp_ratio_pairs = layer_comp_ratio_pairs

    def select(self, layer_db: LayerDatabase, _modules_to_ignore):
        selected_layers = [pair.layer for pair in self._layer_comp_ratio_pairs]
        layer_db.mark_picked_layers(selected_layers)
