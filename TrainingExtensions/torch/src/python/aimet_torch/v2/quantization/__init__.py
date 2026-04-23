# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.quantization

All contents have been moved to aimet_torch.quantization. This module re-exports everything
from the new location for backwards compatibility.
"""

# pylint: disable=wildcard-import, unused-wildcard-import, redefined-builtin
from ...quantization import *
from ...quantization import base, affine, float
from ...quantization.affine import get_backend, set_backend
