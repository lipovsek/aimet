# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.batch_norm_fold.

All contents have been moved to aimet_torch.v2.batch_norm_fold.
This module re-exports everything from the new location for backwards compatibility.
"""

# pylint: disable=unused-import
from ..batch_norm_fold import (
    BatchNormFold,
    fold_all_batch_norms,
    fold_all_batch_norms_to_scale,
    fold_given_batch_norms,
    _is_valid_bn_fold,
    _find_all_batch_norms_to_fold,
    find_standalone_batchnorm_ops,
    find_all_batch_norms_to_fold,
)
