# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Deprecated aimet_common package pointing to aimet_onnx.common or aimet_torch.common
for temporary backward compatibility.

!!DO NOT IMPORT THIS. This package will be deleted in the future version!!
"""

# pylint: disable=redefined-outer-name
import sys
import pkgutil
import importlib
import warnings


def _get_aimet_common_package_name() -> str:
    try:
        import _aimet_common  # Only exists in dev environment
    except ImportError:
        _aimet_common = None

    if _aimet_common:
        return "_aimet_common"

    try:
        import aimet_onnx
    except ImportError:
        aimet_onnx = None

    try:
        import aimet_torch
    except ImportError:
        aimet_torch = None

    if aimet_torch and aimet_onnx:
        raise ImportError(
            "Importing aimet_common failed because both aimet_onnx and aimet_torch are installed. "
            "From aimet v2.20, aimet_common has been deprecated and maintained for backward compatibility only. "
            "aimet_common now points to either aimet_onnx.common or aimet_torch.common, but not both. "
            "To avoid ambiguity, please use one of aimet_onnx.common or aimet_torch.common explicitly."
        )

    if not aimet_onnx and not aimet_torch:
        raise ImportError("Neither aimet_onnx nor aimet_torch was found")

    pkg_name = "aimet_onnx.common" if aimet_onnx else "aimet_torch.common"

    msg = (
        "aimet_common package is deprecated since v2.20 and will be deleted in the future releases. "
        f"Please directly import 👉 {pkg_name} 👈 instead."
    )
    warnings.warn(msg, FutureWarning, stacklevel=3)

    return pkg_name


pkg_name = _get_aimet_common_package_name()
pkg = importlib.import_module(pkg_name)


def register_aimet_common_submodules(pkg):
    """
    Define aimet_common.* as alias to aimet_onnx.common.* or aimet_torch.common.*
    """
    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if (
            modname
            in (
                "libaimet_onnxrt_ops",  # will be imported by aimet_onnx separately
                "AimetEncodingRescaler",  # unused
                "AimetTensorQuantizer",  # will be imported by aimet_torch.common.aimet_tensor_quantizer
            )
        ):
            continue

        module = importlib.import_module(f"{pkg.__name__}.{modname}")
        alias = module.__name__.replace(pkg_name, "aimet_common")

        # Example:
        #   - module: "aimet_onnx.common.quantsim"
        #   - alias: "aimet_common.quantsim"
        sys.modules[alias] = module

        if ispkg:
            register_aimet_common_submodules(module)


register_aimet_common_submodules(pkg)

defs = pkg.defs
connected_graph = pkg.connected_graph
quantsim_config = pkg.quantsim_config
onnx = pkg.onnx
