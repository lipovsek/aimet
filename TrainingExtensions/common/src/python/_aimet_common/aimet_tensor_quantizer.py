# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python3

"""Conditionally imports to use AIMET features using MO and python-only implementations"""

# pylint: disable=unused-wildcard-import, wildcard-import, protected-access
try:
    from .AimetTensorQuantizer import *
except ImportError as err:
    ERROR_MESSAGE = (
        f"AimetTensorQuantizer import failed with the following error:\n\n{err}\n\n"
        "Please check that AimetTensorQuantizer has been built and is compatible with your "
        "current environment."
    )

    class _MetaUnavailableClass(type):
        @classmethod
        def __getattr__(mcs, name):
            raise RuntimeError(
                f"Unable to access attribute {name} of class AimetTensorQuantizer: {ERROR_MESSAGE}"
            )

    class AimetTensorQuantizer(metaclass=_MetaUnavailableClass):
        """
        Placeholder class for raising errors when using this class
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"Unable to initialize class AimetTensorQuantizer: {ERROR_MESSAGE}"
            )

        def __getattr__(self, name):
            raise RuntimeError(
                f"Unable to access attribute {name} of class AimetTensorQuantizer: {ERROR_MESSAGE}"
            )
