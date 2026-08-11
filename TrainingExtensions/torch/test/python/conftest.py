# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import platform
import sys

import pytest
import torch

# torch sizes intra-op/inter-op pools from physical cores, far more than these tests need.
# Aligns with cpu-request in run-unit-acceptance-tests
TORCH_NUM_THREADS = 16

_default_num_threads = torch.get_num_threads()
_default_interop_threads = torch.get_num_interop_threads()
torch.set_num_threads(TORCH_NUM_THREADS)

# Only settable before any inter-op work has started.
try:
    torch.set_num_interop_threads(TORCH_NUM_THREADS)
    _interop_error = None
except RuntimeError as error:
    _interop_error = error


def _thread_config_summary():
    return (
        f"torch intra-op threads: {_default_num_threads} (default) -> {torch.get_num_threads()} (set); "
        f"inter-op threads: {_default_interop_threads} (default) -> {torch.get_num_interop_threads()} (set"
        f"{'' if _interop_error is None else f', failed: {_interop_error}'})"
    )


# Write to stderr
print(_thread_config_summary(), file=sys.stderr)


def _is_macos():
    return sys.platform == "darwin" and platform.machine().lower() in (
        "aarch64",
        "arm64",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_on_macos(reason): skip test on MacOS with specified reason",
    )


@pytest.fixture(autouse=True)
def skip_on_macos(request):
    marker = request.node.get_closest_marker("skip_on_macos")
    if marker is not None:
        if _is_macos():
            reason = marker.args[0] if marker.args else "Not supported on MacOS"
            pytest.skip(reason)


def skip_module_on_macos(reason):
    if _is_macos():
        pytest.skip(allow_module_level=True, reason=reason)
