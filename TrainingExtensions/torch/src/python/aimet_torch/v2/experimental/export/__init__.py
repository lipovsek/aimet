# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from contextlib import nullcontext
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from packaging.version import parse
import torch
from torch.export import ExportedProgram
from ..onnx._export import _precompute_encodings
from ...nn import QuantizationMixin
from ...utils import patch_attr


def export(mod: torch.nn.Module, *args, **kwargs) -> ExportedProgram:
    """
    Export :class:`QuantizationSimModel` to ExportedProgram with
    quantization ops embedded in the aten graph.

    This function takes set of same arguments as `torch.export.export()`_
    """
    if parse(torch.__version__) < parse("2.7"):
        raise RuntimeError(
            "Exporting to torch.exoprt.ExportedProgram is only supported with torch>=2.7; "
            f" got torch=={torch.__version__}"
        )

    with (
        # Pre-compute scale and offset to omit verbose
        # scale/offset derivation logic in the exported graph
        _precompute_encodings(mod),
        # Temporarily suppress parameter monkey patching.
        # Dynamo tracer doesn't work with parameter monkey patching.
        patch_attr(BaseQuantizationMixin, "_patch_quantized_parameters", nullcontext),
        patch_attr(QuantizationMixin, "_patch_dequantized_parameters", nullcontext),
    ):
        return torch.export.export(mod, *args, **kwargs)
