# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.deepspeed_utils

All contents have been moved to aimet_torch.deepspeed_utils. This module re-exports everything
from the new location for backwards compatibility.
"""

# pylint: disable=wildcard-import, unused-wildcard-import, unused-import
from ..deepspeed_utils import (
    SafeGatheredParameters,
    _do_patch_dummy_parameters,
    _ds_ctx,
    _all_gather,
    _patch_dummy_parameters,
    _restore,
    _register_zero3_forward_hooks,
    _shallow_copy,
    _get_shape,
)
