# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.adaround.adaround_weight

All contents have been moved to aimet_torch.adaround.adaround_weight.
This module re-exports everything from the new location for backwards compatibility.
"""

from aimet_torch.adaround.adaround_weight import *  # pylint: disable=wildcard-import, unused-wildcard-import
