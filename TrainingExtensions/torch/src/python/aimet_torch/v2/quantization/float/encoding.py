# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.quantization.float.encoding

All contents have been moved to aimet_torch.quantization.float.encoding. This module re-exports everything
from the new location for backwards compatibility.
"""

# pylint: disable=wildcard-import, unused-wildcard-import
from ....quantization.float.encoding import *
