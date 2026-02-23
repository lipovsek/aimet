# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-docstring

import sys
import sysconfig
import platform

import torch as _torch
from aimet_torch.common import _version
from aimet_torch.common.utils import _red
import warnings


def _is_torch_compatible(current: str, required: str):
    # PyTorch version tag examples:
    #   * 2.1.2+cu121
    #   * 2.1.2+cpu
    #   * 2.1.2
    major, minor, patch = current.split(".")
    required_major, required_minor, required_patch = required.split(".")

    if (major, minor) != (required_major, required_minor):
        return False

    _, *cuda = patch.split("+")
    _, *required_cuda = required_patch.split("+")

    if not cuda or not required_cuda:
        return True

    (cuda,) = cuda
    (required_cuda,) = required_cuda

    # AIMET is always compatible with libtorch unless
    # both AIMET and PyTorch are compiled with CUDA
    if cuda == "cpu" or required_cuda == "cpu":
        return True

    # pylint: disable=unused-variable
    cuda_major, cuda_minor = cuda[:4], cuda[4:]
    required_cuda_major, required_cuda_minor = required_cuda[:4], required_cuda[4:]

    # Only check major CUDA version
    return cuda_major == required_cuda_major


def _is_glibc_compatible(current: str, required: str):
    return list(map(int, required.split("."))) <= list(map(int, current.split(".")))


def _check_requirements():
    reasons = []

    # Check Python ABI.
    # The format of python_abi depends on the current platform.
    # In GNU Linux, it's "{python}-{version}-{arch}-{platform}", where
    #   * python = cpython | ...
    #   * version = 36 | 37 | 38 | 39 | 310 | ...
    #   * arch = x86_64 | aarch64 | ...
    #   * platform = linux-gnu
    python_abi = sysconfig.get_config_var("SOABI")
    if python_abi != _version.python_abi:
        reasons.append(
            f"  * Python: {_version.python_abi} (currently you have {python_abi})"
        )

    # Check PyTorch ABI.
    if not _is_torch_compatible(_torch.__version__, _version.torch):
        major, minor, patch = _version.torch.split(".")
        _, cuda = patch.split("+")
        reasons.append(
            f"  * torch=={major}.{minor}.*+{cuda} "
            f"(currently you have torch=={_torch.__version__})"
        )

    # Check glibc version
    if _version.min_glibc and sys.platform == "linux":
        libc, libc_version = platform.libc_ver()
        if libc != "glibc":
            reasons.append(f"  * libc==glibc (currently you have {libc})")
        elif not _is_glibc_compatible(libc_version, _version.min_glibc):
            reasons.append(
                f"  * glibc>={_version.min_glibc} "
                f"(currently you have glibc=={libc_version})"
            )

    if reasons:
        msg = "\n".join(
            [
                "aimet_torch.v1 package requires following environment:",
                *reasons,
            ]
        )
        raise ImportError(msg)


_check_requirements()

_msg = " ".join(
    [
        "⚠️",
        _red(
            "aimet_torch.v1 is deprecated since aimet 2.0 and will be removed in 2.30. "
            "Please switch to aimet_torch.v2 instead (default since 2.0)",
        ),
        "⚠️",
    ]
)
warnings.warn(_msg, DeprecationWarning, stacklevel=2)
