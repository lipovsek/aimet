# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Export utilities for GenAI testing"""

from datetime import datetime


def get_test_artifacts_path(test_params, base_dir="GenAITests/artifacts/exports"):
    """Generate a deterministic, human-readable artifact directory path."""
    model_id = test_params["model"]["model_id"]
    slug = model_id.split("/")[-1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{base_dir}/{slug}_{timestamp}"
