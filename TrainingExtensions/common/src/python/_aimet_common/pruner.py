# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Creates a compressed model by calling modules to split layers"""

from decimal import Decimal
import abc
from typing import List
import copy

# Import aimet specific modules
from .layer_database import Layer, LayerDatabase
from .defs import CostMetric, LayerCompRatioPair


class Pruner(abc.ABC):
    """
    Models a ML Model Pruner
    """

    def prune_model(
        self,
        layer_db: LayerDatabase,
        layer_comp_ratio_list: List[LayerCompRatioPair],
        cost_metric: CostMetric,
        trainer,
    ) -> LayerDatabase:
        """
        Prune a model given a list of layer-comp_ratio pairs

        :param cost_metric:
        :param layer_db: Layer database of the model to prune
        :param layer_comp_ratio_list: List of layer-comp_ratio pairs
        :param trainer: Used for
        :return: Compressed copy of the LayerDatabase
        """

        # Copy the db
        comp_layer_db = copy.deepcopy(layer_db)
        for layer_comp_ratio in layer_comp_ratio_list:
            layer = comp_layer_db.find_layer_by_name(layer_comp_ratio.layer.name)
            comp_ratio = layer_comp_ratio.comp_ratio

            if comp_ratio is not None and comp_ratio < 1.0:
                self._prune_layer(
                    layer_db, comp_layer_db, layer, comp_ratio, cost_metric
                )

            # fine-tuning the layer while creating the final model
            if trainer is not None:
                trainer.train_model(comp_layer_db.model, layer)

        return comp_layer_db

    @abc.abstractmethod
    def _prune_layer(
        self,
        orig_layer_db: LayerDatabase,
        comp_layer_db: LayerDatabase,
        layer: Layer,
        comp_ratio: Decimal,
        cost_metric: CostMetric,
    ):
        """
        Replaces a given layer within the layer_db with a pruned version of the layer
        In this case, the replaced layer will be a sequential of two spatial-svd-decomposed layers

        :param orig_layer_db: Layer database
        :param comp_layer_db: Compressed layer database, will be modified
        :param layer: Layer to prune
        :param comp_ratio: Compression-ratio
        :param cost_metric: Cost metric
        :return: Actual compression ratio
        """
