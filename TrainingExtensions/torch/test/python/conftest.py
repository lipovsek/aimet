# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import platform
import sys

import pytest


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


def pytest_ignore_collect(collection_path, config):
    """Skip test files that import aimet_torch.v1 on Python 3.11+"""
    if sys.version_info >= (3, 11) and collection_path.suffix == ".py":
        try:
            content = collection_path.read_text()
            if "from aimet_torch.v1" in content or "import aimet_torch.v1" in content:
                return True  # Ignore this file
        except Exception:
            pass
    return False  # Don't ignore


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
