# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Provides an interface for rounding rank or channels"""

import abc
from decimal import Decimal

from .defs import CostMetric
from .layer_database import Layer
from . import utils


class CompRatioRounder(abc.ABC):
    """
    Models a ML Model Pruner
    """

    @abc.abstractmethod
    def round(
        self, layer: Layer, comp_ratio: Decimal, cost_metric: CostMetric
    ) -> Decimal:
        """
        Rounds comp_ratio to nearest multiplicity
        :param layer: Layer
        :param comp_ratio: comp_ratio to be rounded
        :param cost_metric:
        :return: Updated comp_ratio
        """


class RankRounder(CompRatioRounder):
    """
    Rounds rank and finds the corresponding updated compression ratio for SVD
    """

    def __init__(self, multiplicity: int, cost_calculator):
        """
        :param multiplicity: Multiplicity to which rank is rounded-up
        :param cost_calculator: SpatialSvdCostCalculator() or WeightSvdCostCalculator()
        """
        self._multiplicity = multiplicity
        self._cost_calculator = cost_calculator

    def round(
        self, layer: Layer, comp_ratio: Decimal, cost_metric: CostMetric
    ) -> Decimal:
        if self._multiplicity == 1:
            return comp_ratio

        # Find rank corresponding to a compression ratio
        rank = self._cost_calculator.calculate_rank_given_comp_ratio(
            layer, comp_ratio, cost_metric
        )

        # Finding rank corresponding to comp ratio 1
        max_rank = self._cost_calculator.calculate_rank_given_comp_ratio(
            layer, 1.0, cost_metric
        )

        rounded_rank_candidate = utils.round_up_to_multiplicity(
            self._multiplicity, rank, max_rank
        )
        if rank == rounded_rank_candidate:
            updated_comp_ratio = comp_ratio
        else:
            # For the rounded rank compute the new compression ratio
            updated_comp_ratio = self._cost_calculator.calculate_comp_ratio_given_rank(
                layer, rounded_rank_candidate, cost_metric
            )

        assert 0 <= updated_comp_ratio <= 1
        assert comp_ratio <= updated_comp_ratio

        return updated_comp_ratio


class ChannelRounder(CompRatioRounder):
    """
    Rounds input channels to be kept and finds the corresponding updated compression ratio for Channel Pruning
    """

    def __init__(self, multiplicity: int):
        """
        :param multiplicity: Multiplicity to which rank is rounded-up
        """
        self._multiplicity = multiplicity

    def round(
        self, layer: Layer, comp_ratio: Decimal, cost_metric: CostMetric
    ) -> Decimal:
        if self._multiplicity == 1:
            updated_comp_ratio = comp_ratio
        else:
            # get number of input channels to keep
            in_channels = layer.weight_shape[1]
            keep_inp_channels = in_channels * comp_ratio
            # Round input channels
            keep_inp_channels = utils.round_up_to_multiplicity(
                self._multiplicity, keep_inp_channels, in_channels
            )

            # Find updated compression ratio
            updated_comp_ratio = Decimal(keep_inp_channels) / Decimal(in_channels)
            assert comp_ratio <= updated_comp_ratio
            assert 0 <= updated_comp_ratio <= 1

        return updated_comp_ratio
