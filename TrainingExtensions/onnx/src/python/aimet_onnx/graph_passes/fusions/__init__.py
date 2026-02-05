# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX graph fusion passes"""

from .fusion import fuse_supergroups
from .fusion_registry import AIMET_SUPERGROUP_DOMAIN
from .layernorm import LayerNormFusion
