# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Prunes layers using SpatialSvdModuleSplitter SVD scheme"""

import copy

from aimet_torch.common.utils import AimetLogger
from aimet_torch.common.defs import CostMetric
from aimet_torch.common import cost_calculator
from aimet_torch.common import svd_pruner

from aimet_torch.svd.svd_splitter import (
    SpatialSvdModuleSplitter,
    WeightSvdModuleSplitter,
)
from aimet_torch.layer_database import LayerDatabase, Layer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdPruner(svd_pruner.SpatialSvdPruner):
    """
    Pruner for Spatial-SVD method
    """

    def _perform_svd_and_split_layer(
        self, layer: Layer, rank: int, comp_layer_db: LayerDatabase
    ):
        """
        Performs spatial svd and splits given layer into two layers
        :param layer: Layer to split
        :param rank: Rank to use for spatial svd splitting
        :param comp_layer_db: Compressed layer db to update with the split layers
        :return: None
        """
        # Split module using Spatial SVD
        module_a, module_b = SpatialSvdModuleSplitter.split_module(layer.module, rank)

        first_layer_shape = copy.copy(layer.output_shape)

        # Set the second dimension (output_channels) to the rank being used for splitting first layer
        first_layer_shape[1] = rank
        # Special logic for strided layers
        first_layer_shape[3] = first_layer_shape[3] * layer.module.stride[1]

        # Create two new layers and return them
        layer_a = Layer(module_a, layer.name + ".0", first_layer_shape)
        layer_b = Layer(module_b, layer.name + ".1", layer.output_shape)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(
            layer, layer_a, layer_b
        )


class WeightSvdPruner(svd_pruner.WeightSvdPruner):
    """
    Pruner for Weight-SVD method using numpy.
    """

    # pylint: disable=no-self-use
    def _perform_svd_and_split_layer(
        self,
        layer: Layer,
        rank: int,
        cost_metric: CostMetric,
        comp_layer_db: LayerDatabase,
    ):
        """
        Performs spatial svd and splits given layer into two layers
        :param layer: Layer to split
        :param rank: Rank to use for weight svd splitting
        :param comp_layer_db: Compressed layer db to update with the split layers
        :return: None
        """
        # For the rounded rank compute the new compression ratio
        comp_ratio = (
            cost_calculator.WeightSvdCostCalculator.calculate_comp_ratio_given_rank(
                layer, rank, cost_metric
            )
        )
        # Split module using Weight SVD
        logger.info("Splitting module: %s with rank: %r", layer.name, rank)
        module_a, module_b = WeightSvdModuleSplitter.split_module(layer.module, rank)

        layer_a = Layer(module_a, layer.name + ".0", layer.output_shape)
        layer_b = Layer(module_b, layer.name + ".1", layer.output_shape)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(
            layer, layer_a, layer_b
        )

        return comp_ratio
