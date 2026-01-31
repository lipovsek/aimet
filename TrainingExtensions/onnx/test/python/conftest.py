# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import platform
import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_on_windows_arm64(reason): skip test on Windows ARM64 with specified reason",
    )


def _is_windows_arm64():
    return sys.platform == "win32" and platform.machine().lower() in (
        "aarch64",
        "arm64",
    )


@pytest.fixture(autouse=True)
def skip_on_windows_arm64(request):
    marker = request.node.get_closest_marker("skip_on_windows_arm64")
    if marker is not None:
        if _is_windows_arm64():
            reason = marker.args[0] if marker.args else "Not supported on Windows ARM64"
            pytest.skip(reason)


def skip_module_on_windows_arm64(reason):
    """Helper for module-level skips. Call at top of test module."""
    if _is_windows_arm64():
        pytest.skip(allow_module_level=True, reason=reason)
