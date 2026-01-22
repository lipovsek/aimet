# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=wildcard-import, unused-wildcard-import, unused-import
"""Alias to aimet_torch._base.amp.mixed_precision_algo"""

from .._base.amp.mixed_precision_algo import *
from .._base.amp.mixed_precision_algo import (
    _default_forward_fn,
    _compute_sqnr,
    _evaluate_sqnr,
)
