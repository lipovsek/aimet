# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.quantsim.quantsim

All contents have been moved to aimet_torch.quantsim.quantsim.
This module re-exports everything from the new location for backwards compatibility.
"""

from aimet_torch.quantsim.quantsim import *  # pylint: disable=wildcard-import, unused-wildcard-import
