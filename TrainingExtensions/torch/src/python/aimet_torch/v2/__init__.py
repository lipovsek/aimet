# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Backwards compatibility shim for aimet_torch.v2

All contents have been moved to aimet_torch. This module re-exports everything
from the new location for backwards compatibility.
"""

from . import experimental
from . import nn
from . import quantization
from . import quantsim
from . import utils
from . import visualization_tools

import warnings

warnings.warn(
    "All modules in `aimet_torch.v2` subpackage have been moved to top-level `aimet_torch` from v2.30. "
    "`aimet_torch.v2` only remains as a placeholder for backward compatibility. "
    "Import from `aimet_torch` instead of `aimet_torch.v2` to avoid this warning",
    DeprecationWarning,
    stacklevel=2,
)
