# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from packaging.version import parse
import torch
from torch.export import ExportedProgram
from ..onnx._export import _precompute_encodings
from ...nn import QuantizationMixin


def export(mod: torch.nn.Module, *args, **kwargs) -> ExportedProgram:
    """
    Export :class:`QuantizationSimModel` to ExportedProgram with
    quantization ops embedded in the aten graph.

    This function takes set of same arguments as `torch.export.export()`_
    """
    # pylint: disable=protected-access
    if parse(torch.__version__) < parse("2.7"):
        raise RuntimeError(
            "Exporting to torch.exoprt.ExportedProgram is only supported with torch>=2.7; "
            f" got torch=={torch.__version__}"
        )

    untraceable_modules = {
        name: module
        for name, module in mod.named_modules()
        if isinstance(module, QuantizationMixin) and not module._is_dynamo_traceable()
    }

    if untraceable_modules:
        raise RuntimeError(
            "Following modules don't support dynamo tracing:\n"
            + "\n".join(
                [
                    f"- {name} (type: {type(module).__name__})"
                    for name, module in untraceable_modules.items()
                ]
            )
        )

    # Pre-compute scale and offset to omit verbose
    # scale/offset derivation logic in the exported graph
    with _precompute_encodings(mod), torch.no_grad():
        return torch.export.export(mod, *args, **kwargs)
