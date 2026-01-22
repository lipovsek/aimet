# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Functions for validating pytorch models prior to using AIMET features"""

from typing import Tuple, Union, Set, List
import torch

from aimet_torch.common.utils import AimetLogger
from aimet_torch import utils
from aimet_torch.meta import connectedgraph_utils
from aimet_torch.meta.operation import Op


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def validate_for_reused_modules(
    model: torch.nn.Module,
    model_input: Union[torch.Tensor, Tuple],
    layers_to_exclude: Union[List[torch.nn.Module], None] = None,
    **kwargs,
) -> bool:
    """
    Check if the model has any reused modules. Returns True if there are none, False otherwise.
    :param model: Pytorch model to create connected graph from
    :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
    :param layers_to_exclude: List of layers to exclude from checking for reused modules
    :return: True if model has no reused modules, False otherwise
    """
    # pylint: disable=unused-argument
    if not layers_to_exclude:
        layers_to_exclude = []

    layers_to_exclude = set(layers_to_exclude)
    blacklisted_layers = _get_blacklisted_layers(layers_to_exclude)
    reused_modules = utils.get_reused_modules(model, model_input)
    reused_modules = [
        (name, module)
        for (name, module) in reused_modules
        if module not in blacklisted_layers
    ]
    if reused_modules:
        logger.error(
            "The following modules are used more than once in the model: %s\n"
            "AIMET features are not designed to work with reused modules. Please redefine your model "
            "to use distinct modules for each instance.",
            [name for (name, _) in reused_modules],
        )
    return not reused_modules


def validate_for_missing_modules(
    model: torch.nn.Module,
    model_input: Union[torch.Tensor, Tuple],
    layers_to_exclude: Union[List[torch.nn.Module], None] = None,
    **kwargs,
) -> bool:
    """
    Check if the model has any ops with missing modules (excluding a set of known ops which can be functionals).
    Returns True if there are no ops with missing modules, False otherwise.
    :param model: Pytorch model to create connected graph from
    :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
    :param layers_to_exclude: List of layers to exclude from checking for missing modules
    :return: True if model has no ops with missing modules, False otherwise.
    """
    # pylint: disable=unused-argument
    if not layers_to_exclude:
        layers_to_exclude = []
    filtered_ops_with_missing_modules = _get_filtered_ops_with_missing_modules(
        model, model_input, layers_to_exclude
    )

    # pylint: disable=protected-access
    module_to_name_dict = utils.get_module_to_name_dict(model, prefix=model._get_name())

    if filtered_ops_with_missing_modules:
        # TODO: replace with logger.error and assertion after rewriting unit tests to avoid using built in vgg,
        #  resnet, and inception models (since they use functionals in their models)
        error_message = (
            "Functional ops that are not marked as math invariant were found in the model. AIMET features "
            "will not work properly for such ops.\n"
            "Consider the following choices: \n"
            "1. Redefine as a torch.nn.Module in the class definition.\n"
            "2. The op can remain as a functional op due to being math invariant, but the op type has not "
            "been added to ConnectedGraph.math_invariant_types set. \n"
            "Add an entry to ignore the op by importing the set and adding the op type:\n\n"
            "\tfrom aimet_torch.meta.connectedgraph import ConnectedGraph\n"
            "\tConnectedGraph.math_invariant_types.add(...)\n\n"
            "The following functional ops were found. The parent module is named for ease of "
            "locating the ops within the model definition.\n"
        )
        max_name_len = 0
        for op in filtered_ops_with_missing_modules:
            if len(op.name) + len(op.type) > max_name_len:
                max_name_len = len(op.name) + len(op.type)
        for op in filtered_ops_with_missing_modules:
            error_message += (
                f"\t{op.name}; op_type: {op.type}{' ' * (max_name_len + 10 - (len(op.name) + len(op.type)))}"
                f"parent module: {module_to_name_dict.get(op.residing_module)}\n"
            )
        logger.error(error_message)
    return not filtered_ops_with_missing_modules


def _get_filtered_ops_with_missing_modules(
    model: torch.nn.Module,
    model_input: Union[torch.Tensor, Tuple],
    layers_to_exclude: List[torch.nn.Module],
) -> List[Op]:
    """
    Get a list of ops with missing modules excluding ones of types in excluded_layer_types, as well as ones residing
    within layers whose types appear in excluded_layer_types.
    :param model: Torch model to get ops with missing modules for
    :param model_input: Dummy input to the torch model
    :param layers_to_exclude: List of layers to exclude looking for ops with missing modules in
    :return: List of filtered ops with missing modules
    """
    layers_to_exclude = set(layers_to_exclude)
    blacklisted_layers = _get_blacklisted_layers(layers_to_exclude)
    ops_with_missing_modules = connectedgraph_utils.get_ops_with_missing_modules(
        model, model_input
    )
    filtered_ops_with_missing_modules = [
        op
        for op in ops_with_missing_modules
        if op.residing_module not in blacklisted_layers
    ]
    return filtered_ops_with_missing_modules


def _get_blacklisted_layers(
    layers_to_exclude: Set[torch.nn.Module],
) -> Set[torch.nn.Module]:
    """
    Get a set of modules consisting of layers to exclude and their submodules.
    :param layers_to_exclude: Set of layers to exclude
    :return: Set of excluded layers and their submodules.
    """
    blacklisted_layers = set()
    for layer in layers_to_exclude:
        blacklisted_layers.add(layer)
        for module in layer.modules():
            blacklisted_layers.add(module)
    return blacklisted_layers
