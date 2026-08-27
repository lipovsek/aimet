# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Dict
from aimet_onnx import qtype, int16, float16, QuantizationSimModel
from aimet_onnx.common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def flip_layers_to_higher_precision(
    sim: QuantizationSimModel,
    layer_sensitivity_dict: Dict[str, float],
    percent_to_flip: float = 10.0,
    override_precision: qtype | tuple[qtype, qtype] = float16,
):
    """
    Given a sim object and a layer-sensitivity dictionary, flip a given percentage of the layers to higher precision.

    :param sim: QuantizationSimModel instance initialized with the base precision
    :param layer_sensitivity_dict: Dict of (layer_name: sqnr_metric) that is output from analyze_per_layer_sensitivity
    :param percent_to_flip: Percentage of layers to flip
    :param override_precision: Precision to set layers to. Supports single precision for parameters and activations,
        or tuple of (param_type, activation_type).
    """

    if isinstance(override_precision, (tuple, list)):
        if len(override_precision) != 2:
            raise ValueError(
                "Override precision must be a single qtype or tuple of "
                f"(param_type, activation_type), got {override_precision}"
            )

        param_type, activation_type = override_precision
    else:
        param_type = activation_type = override_precision

    param_type = qtype.as_qtype(param_type)
    activation_type = qtype.as_qtype(activation_type)

    if activation_type not in (int16, float16):
        raise ValueError("Activation override_precision must be int16 or float16")

    sqnr_list = sorted(layer_sensitivity_dict.items(), key=lambda item: item[1])
    sqnr_list = sqnr_list[: math.ceil(len(sqnr_list) * percent_to_flip / 100)]
    cg_ops = sim.connected_graph.get_all_ops()

    layer_names_to_override = [layer_name for layer_name, _ in sqnr_list]
    logger.info(
        "Overriding the following layers to precision W-%s A-%s: %s",
        param_type,
        activation_type,
        layer_names_to_override,
    )

    for layer_name in layer_names_to_override:
        op = cg_ops[layer_name]
        (
            input_quantizers,
            output_quantizers,
            param_quantizers,
        ) = sim.get_op_quantizers(op)
        for q in input_quantizers + output_quantizers:
            q.set_precision(activation_type)

        for _, q in param_quantizers.items():
            q.set_precision(param_type)

    sim._apply_exception_rules()  # pylint: disable=protected-access
