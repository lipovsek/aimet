# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""
Backwards compatibility shim for aimet_torch.v2.cg_utils

All contents have been moved to aimet_torch.cg_utils. This module re-exports everything
from the new location for backwards compatibility.
"""

# pylint: disable=wildcard-import, unused-wildcard-import
from ..cg_utils import *
