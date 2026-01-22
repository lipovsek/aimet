# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python3

"""Conditionally imports to use AIMET features using MO and python-only implementations"""

# pylint: disable=unused-wildcard-import, wildcard-import, protected-access
try:
    from ._libpymo import *
except ImportError as err:
    from . import py_libpymo

    py_libpymo.IMPORT_ERROR = err
    from .py_libpymo import *
