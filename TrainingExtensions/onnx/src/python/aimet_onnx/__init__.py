# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-module-docstring

from aimet_onnx.common.defs import qtype, int2, int4, int8, int16, float16

try:
    from aimet_onnx.common import _version

    __version__ = _version.__version__
except ImportError:
    # For convenience: This enables importing aimet_torch from source
    # without building aimet_common._version
    __version__ = None

from .quantsim import QuantizationSimModel, compute_encodings
from .adaround.adaround_weight import apply_adaround
from .sequential_mse.seq_mse import apply_seq_mse
from .quant_analyzer import analyze_per_layer_sensitivity
