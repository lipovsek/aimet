# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Export utilities for GenAI testing"""

import uuid


def get_test_artifacts_path(test_params):
    # todo: change this to something else, based on hashing test params
    return f"artifacts/{uuid.uuid4().hex[-10:]}"
