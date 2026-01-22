# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Per layer cost calculator specialized for PyTorch"""

from typing import List
import numpy as np

import torch

from aimet_torch.common.cost_calculator import (
    CostCalculator as GenericCostCalculator,
    Cost,
)
from aimet_torch.layer_database import Layer
from aimet_torch._base.nn.modules.custom import Multiply

SUPPORTED_LAYER_TYPES = [
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.activation.Sigmoid,
    torch.nn.modules.pooling.AvgPool2d,
    Multiply,
]


class CostCalculator(GenericCostCalculator):
    """This is a specialized implementation of CostCalculator for PyTorch"""

    @staticmethod
    def compute_layer_cost(layer: Layer):
        """
        Computes per layer cost. This is a specialized function for PyTorch.

        :param layer: Attributes for a layer
        :return: Cost of the layer
        """

        assert isinstance(layer.module, tuple(SUPPORTED_LAYER_TYPES)), (
            "Unsupported layer type encountered"
        )

        weight_dim = list(layer.weight_shape)
        if len(weight_dim) > 1:
            additional_act_dim = [layer.output_shape[-1], layer.output_shape[-2]]
            mem_cost = np.prod(weight_dim).item()
            mac_dim = weight_dim + additional_act_dim
            mac_cost = np.prod(mac_dim).item()

        else:  # For Ops w/o weights (or single dim), take the max b/w input and output size to calculate cost
            output_mac_cost = np.prod(layer.output_shape).item()
            input_mac_cost = 0
            if layer.input_shape:
                if not isinstance(layer.input_shape, List):
                    layer.input_shape = [layer.input_shape]
                input_mac_cost = max(
                    np.prod(i_input).item() for i_input in layer.input_shape
                )

            mac_cost = max([output_mac_cost, input_mac_cost])
            mem_cost = mac_cost

        return Cost(mem_cost, mac_cost)
