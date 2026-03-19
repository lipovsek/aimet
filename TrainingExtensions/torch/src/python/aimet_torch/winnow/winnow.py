# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Search a graph for winnowing opportunities and apply all required changes."""

from typing import List, Tuple
import torch
from aimet_torch.common.utils import AimetLogger, deprecated
from aimet_torch.winnow.mask_propagation_winnower import MaskPropagationWinnower

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


@deprecated(deletion_planned="v2.31.0")
def winnow_model(
    model: torch.nn.Module,
    input_shape: Tuple,
    list_of_modules_to_winnow: List[Tuple[torch.nn.Module, List]] = None,
    reshape=True,
    in_place=False,
    verbose=False,
):
    """This API is used to winnow a model with Conv2d modules that each have a list of channels to be winnowed.
    There is no need to zero out the modules' input channels before calling this API.

    :param model: The model to be winnowed.
    :param input_shape: The input shape of the model.
    :param list_of_modules_to_winnow: A list of Tuples with each Tuple containing a module and a list of channels
                                             to be winnowed for that module.
    :param reshape: If set to True a Down Sample Layer is added between modules to match the number of channels.
                    If set to False, the modules that need a Down Sample Layer will not be winnowed.
    :param in_place: If set to True, the model will be winnowed in place.
                     If set to False, a copy of the model will be winnowed.
    :param verbose: If set to True, logs detailed winnowing log messages.
    :return: If winnowing is successful, a winnowed model is returned. Otherwise, returns None.
    """

    mask_winnower = MaskPropagationWinnower(
        model, input_shape, list_of_modules_to_winnow, reshape, in_place, verbose
    )
    new_model, ordered_modules_list = mask_winnower.propagate_masks_and_winnow()

    return new_model, ordered_modules_list
