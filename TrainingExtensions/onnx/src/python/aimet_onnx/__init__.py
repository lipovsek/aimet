# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-module-docstring

from aimet_onnx.common.defs import qtype, int2, int4, int8, int16, float16
from aimet_onnx.common.utils import _get_version_string

__version__ = _get_version_string()
del _get_version_string

from .quantsim import QuantizationSimModel, compute_encodings
from .adaround.adaround_weight import apply_adaround
from .sequential_mse.seq_mse import apply_seq_mse
from .quant_analyzer import analyze_per_layer_sensitivity
from .defs import QSpec
