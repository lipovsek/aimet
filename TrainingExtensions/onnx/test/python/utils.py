# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def tmp_dir():
    """
    Pytest fixture to create and yield a temporary directory.
    The directory is automatically cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def add_genai_tests_path(monkeypatch):
    """
    Pytest fixture to add the GenAILab directory to sys.path.
    """
    path = os.path.abspath(os.path.join(Path(__file__).parent, "../../../../"))
    monkeypatch.syspath_prepend(path)


@contextmanager
def patch_fuse_supergroups(enabled: bool):
    """
    Context manager that patches ``aimet_onnx.quantsim._fuse_supergroups`` to the
    given value for the duration of the block.
    """
    with patch("aimet_onnx.quantsim._fuse_supergroups", enabled):
        yield
