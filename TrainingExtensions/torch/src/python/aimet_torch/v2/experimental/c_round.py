# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-module-docstring
import torch

from ..quantization.affine.backends.torch_builtins import _set_round_fn


def c_round(tensor: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Applies c-style rounding (std::round).

    Note that c-style rounding rounds the halfway cases away from zero,
    unlike python-, numpy-, or pytorch-style rounding which rounds halfway
    cases to the nearest even integer.

    Examples:

        >>> x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        >>> torch.round(x)
        tensor([-2., -2., -0.,  0.,  2.,  2.])
        >>> c_round(x)
        tensor([-3., -2., -1.,  1.,  2.,  3.])
    """
    tensor = torch.add(tensor, tensor.sign(), alpha=0.5, out=out)
    return torch.trunc(tensor, out=out)


def c_round_(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place version of c_round
    """
    return c_round(tensor, out=tensor)


def use_c_round(flag: bool):
    """
    Let AIMET quantziers use c-style rounding (std::round) under the hook.

    C-style rounding rounds the halfway cases away from zero,
    unlike python-, numpy-, or pytorch-style rounding which rounds halfway
    cases to the nearest even integer.

    Examples:

        >>> x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        >>> torch.round(x)
        tensor([-2., -2., -0.,  0.,  2.,  2.])
        >>> c_round(x)
        tensor([-3., -2., -1.,  1.,  2.,  3.])


    :param flag: If True, use c-style rounding. Otherwise, use default torch.round.
    """
    if flag:
        _set_round_fn(c_round, c_round_)
    else:
        _set_round_fn(torch.round, torch.round_)
