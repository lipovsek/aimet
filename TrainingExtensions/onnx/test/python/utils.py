# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import tempfile
from pathlib import Path


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
