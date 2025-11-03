# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Dict
from aimet_onnx import qtype, int16, float16, QuantizationSimModel


def flip_layers_to_higher_precision(
    sim: QuantizationSimModel,
    layer_sensitivity_dict: Dict[str, float],
    percent_to_flip: float = 10.0,
    override_precision: qtype = float16,
):
    """
    Given a sim object and a layer-sensitivity dictionary, flip a given percentage of the layers to higher precision.

    :param sim: QuantizationSimModel instance initialized with the base precision
    :param layer_sensitivity_dict: Dict of (layer_name: sqnr_metric) that is output from analyze_per_layer_sensitivity
    :param percent_to_flip: Percentage of layers to flip
    :param override_precision: Precision to sets layers to. At present, either int16 (w16a16) or float16 are supported.
    """

    # Sanity check
    if override_precision not in (int16, float16):
        raise ValueError("higher_precision must be int16 or float16")

    sqnr_list = sorted(layer_sensitivity_dict.items(), key=lambda item: item[1])
    sqnr_list = sqnr_list[: math.ceil(len(sqnr_list) * percent_to_flip / 100)]
    cg_ops = sim.connected_graph.get_all_ops()

    for layer_name, _ in sqnr_list:
        op = cg_ops[layer_name]
        (
            input_quantizers,
            output_quantizers,
            param_quantizers,
        ) = sim.get_op_quantizers(op)
        for q in input_quantizers + output_quantizers:
            if override_precision == int16:
                q.set_bitwidth(16)
            else:
                q.enabled = False

        for _, q in param_quantizers.items():
            if override_precision == int16:
                q.set_bitwidth(16)
            else:
                q.enabled = False
