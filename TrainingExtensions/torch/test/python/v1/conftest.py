# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Skip all v1 tests on Python 3.11+ since aimet_torch.v1 requires Python 3.10
"""

import sys

# Skip all tests in this directory on Python 3.11+
if sys.version_info >= (3, 11):
    collect_ignore_glob = ["*.py"]
