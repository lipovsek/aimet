# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""This file contains utilities needed to support pytorch nn transformer layer"""

import torch
from aimet_torch import utils
from aimet_torch.model_preparer import prepare_pt_transformer_for_quantsim
from aimet_torch.transformers.activation import create_quantizable_multihead_attention
from aimet_torch.common.utils import deprecated


@deprecated(deletion_planned="v2.31.0")
def get_quantizable_pt_transformer_model(model: torch.nn.Module):
    """
    This auto replaces pt MHA with Quantizable version
    Also, replaces act fn functionals with  modules
    :param: model : Input model with PT transformer layer
    :return: updates model in-place, as necessary.
    """
    # auto replace PyTorch MHA in given transformer layer with quantizable MHA
    utils.replace_modules(
        model,
        lambda module: isinstance(module, torch.nn.MultiheadAttention),
        create_quantizable_multihead_attention,
    )

    # auto replace functional activation with module for nn.Transformer layers
    prepare_pt_transformer_for_quantsim(model)
