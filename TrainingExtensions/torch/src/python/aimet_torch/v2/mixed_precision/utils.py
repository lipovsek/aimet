# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Backwards compatibility shim for aimet_torch.v2.mixed_precision.utils

All contents have been moved to aimet_torch.mixed_precision.utils
This module re-exports everything from the new location for backwards compatibility.
"""

# pylint: disable=wildcard-import, unused-wildcard-import
from aimet_torch.mixed_precision.utils import *
