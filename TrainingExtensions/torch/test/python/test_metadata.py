# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import importlib.metadata
import pytest
import aimet_torch


try:
    metadata = importlib.metadata.metadata("aimet-torch")
except importlib.metadata.PackageNotFoundError:
    pytest.skip(allow_module_level=True)


def test_metadata():
    """
    When: import aimet_torch
    Then: __version__ string should be consistent with what is specified in package metadata
    """
    assert aimet_torch.__version__ == metadata["version"]
