# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.nn.true_quant

All contents have been moved to aimet_torch.nn.true_quant. This module re-exports everything
from the new location for backwards compatibility.
"""

from ...nn.true_quant import *  # pylint: disable=wildcard-import, unused-wildcard-import
