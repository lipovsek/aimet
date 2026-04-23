# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2.nn.transformers.models.llama.modeling_llama

All contents have been moved to aimet_torch.nn.transformers.models.llama.modeling_llama. This module re-exports everything
from the new location for backwards compatibility.
"""

from ......nn.transformers.models.llama.modeling_llama import *  # pylint: disable=wildcard-import, unused-wildcard-import
