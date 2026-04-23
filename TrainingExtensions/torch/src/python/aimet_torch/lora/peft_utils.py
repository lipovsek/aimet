# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Implementation for handling LoRA adapters added using PEFT"""

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.nn import BaseQuantizationMixin, lora as qlora


def _get_lora_layer_except_base_layer(sim: QuantizationSimModel):
    part_of_lora_layer_except_base = set()
    for module in sim.model.modules():
        if isinstance(module, (qlora.QuantizedLinear, qlora.QuantizedConv)):
            for m in module.modules():
                if isinstance(m, BaseQuantizationMixin) and m != module.base_layer:
                    part_of_lora_layer_except_base.add(m)
    return part_of_lora_layer_except_base


def _freeze_quantizer(quantizer):
    """
    Disables compute encodings and gradient update for a quantizer

    :param quantizer: Param, output or Input quantizer
    """
    # pylint:disable = protected-access
    quantizer._allow_overwrite = False
    quantizer.requires_grad_(False)


def freeze_base_model_param_quantizers(sim: QuantizationSimModel):
    """
    Freeze parameter quantizers of base model

    :param sim: QuantSim model
    """

    def _freeze(module):
        for param_quantizer in module.param_quantizers.values():
            if param_quantizer:
                _freeze_quantizer(param_quantizer)

    part_of_lora_layer_except_base = _get_lora_layer_except_base_layer(sim)
    for module in sim.model.modules():
        if (
            isinstance(module, BaseQuantizationMixin)
            and module not in part_of_lora_layer_except_base
        ):
            _freeze(module)


def freeze_base_model_activation_quantizers(sim: QuantizationSimModel):
    """
    Freeze activation quantizers of base model

    :param sim: QuantSim model
    """

    def _freeze(module):
        for input_quantizer, output_quantizer in zip(
            module.input_quantizers, module.output_quantizers
        ):
            if input_quantizer:
                _freeze_quantizer(input_quantizer)
            if output_quantizer:
                _freeze_quantizer(output_quantizer)

    part_of_lora_layer_except_base = _get_lora_layer_except_base_layer(sim)
    for module in sim.model.modules():
        if (
            isinstance(module, BaseQuantizationMixin)
            and module not in part_of_lora_layer_except_base
        ):
            _freeze(module)


def freeze_base_model(sim: QuantizationSimModel):
    """
    Freeze entire base model

    :param sim: QuantSim model
    """
    freeze_base_model_param_quantizers(sim)
    freeze_base_model_activation_quantizers(sim)
