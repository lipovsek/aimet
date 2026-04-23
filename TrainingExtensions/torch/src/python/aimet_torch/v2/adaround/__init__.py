# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.adaround

All contents have been moved to aimet_torch.adaround.
This module re-exports everything from the new location for backwards compatibility.
"""

from aimet_torch.adaround import (
    adaround_weight,
    Adaround,
    AdaroundWrapper,
    AdaroundParameters,
)
