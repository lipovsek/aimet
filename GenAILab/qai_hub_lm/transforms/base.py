# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Base classes and utilities for model adaptations."""

import functools
from abc import abstractmethod, ABC


class AdaptedModule(ABC):
    """Base class for modules that require post-instantiation adaptation."""

    @abstractmethod
    def adapt(self):
        """Apply the adaptation to this module."""
        pass


def rgetattr(obj, attr, *args):
    """Recursive getattr for nested attributes (e.g., 'model.layer.weight')."""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    """Recursive setattr for nested attributes (e.g., 'model.layer.weight')."""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
