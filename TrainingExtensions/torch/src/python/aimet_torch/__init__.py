# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-module-docstring

try:
    from aimet_torch.common import _version

    __version__ = _version.__version__
except ImportError:
    # For convenience: This enables importing aimet_torch from source
    # without building aimet_common._version
    __version__ = None

from .quantsim import QuantizationSimModel
from . import nn
from . import quantization
from . import onnx
