# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Compatibility module for onnxscript API changes between 0.3.x and 0.4.x.

In onnxscript >= 0.4.0, several private symbols were promoted to public API:
  - ``pattern.Var``: was ``_pattern_ir.ValuePattern``
  - ``pattern.AttrVar`: was ``_pattern_ir.AttrPattern``
"""

# pylint: disable = unused-import
try:
    from onnxscript.rewriter.pattern import Var, AttrVar  # onnxscript >= 0.4.0

except ImportError:
    from onnxscript.rewriter._pattern_ir import Var, AttrPattern

    AttrVar = AttrPattern
