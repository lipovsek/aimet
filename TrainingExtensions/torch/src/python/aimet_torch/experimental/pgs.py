# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""APIs for enabling/disabling Pivot Gradient Spiking (PGS)"""

__all__ = [
    "enable_pgs",
    "disable_pgs",
    "is_pgs_enabled",
    "get_pgs_eps",
    "get_pgs_multiplier",
]

_PGS_EPS: float = 0.0
_PGS_MULTIPLIER: float = 1.0


def enable_pgs(eps: float = 0.05, multiplier: float = 3.0):
    global _PGS_EPS, _PGS_MULTIPLIER  # pylint: disable=global-statement

    if not (0.0 < eps <= 1.0):
        raise ValueError(f"eps must be ∈ (0, 1] to enable PGS. Got {eps}")

    if multiplier == 1.0:
        raise ValueError("multiplier must be ≠1 to enable PGS")

    _PGS_EPS = eps
    _PGS_MULTIPLIER = multiplier


def disable_pgs():
    global _PGS_EPS, _PGS_MULTIPLIER  # pylint: disable=global-statement
    _PGS_EPS = 0.0
    _PGS_MULTIPLIER = 1.0


def is_pgs_enabled() -> bool:
    return _PGS_EPS > 0.0 and _PGS_MULTIPLIER != 1.0


def get_pgs_eps() -> float:
    return _PGS_EPS


def get_pgs_multiplier() -> float:
    return _PGS_MULTIPLIER
