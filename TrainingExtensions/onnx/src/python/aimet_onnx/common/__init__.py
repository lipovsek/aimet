# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Placeholder aimet_onnx.common subpackage pointing to _aimet_common for development environment.
This subpackage will be substituted with _aimet_common in production environment.
"""

import sys
import pkgutil
import importlib
import _aimet_common


def register_aimet_common_submodules(pkg):
    """
    Define aimet_onnx.common.* as alias to _aimet_common.*
    """
    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if (
            modname == "libaimet_onnxrt_ops"
        ):  # will be imported by aimet_onnx separately
            continue

        module = importlib.import_module(f"{pkg.__name__}.{modname}")
        alias = module.__name__.replace("_aimet_common", "aimet_onnx.common")

        # Example:
        #   - module: "_aimet_common.quantsim"
        #   - alias: "aimet_onnx.common.quantsim"
        sys.modules[alias] = module

        if ispkg:
            register_aimet_common_submodules(module)


register_aimet_common_submodules(_aimet_common)

defs = _aimet_common.defs
connected_graph = _aimet_common.connected_graph
quantsim_config = _aimet_common.quantsim_config
onnx = _aimet_common.onnx
