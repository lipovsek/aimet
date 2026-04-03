# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Skip tests that import aimet_torch.v1 on Python 3.11+ since aimet_torch.v1 requires Python 3.10
"""

import sys


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
