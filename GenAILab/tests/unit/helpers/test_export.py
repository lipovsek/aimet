# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for export utilities."""

import re

from GenAILab.bench.export import get_test_artifacts_path


class TestGetTestArtifactsPath:
    def test_slug_from_model_id(self):
        params = {"model": {"model_id": "meta-llama/Llama-3.2-1B-Instruct"}}
        path = get_test_artifacts_path(params)
        assert "Llama-3.2-1B-Instruct" in path

    def test_custom_base_dir(self):
        params = {"model": {"model_id": "meta-llama/Llama-3.2-1B-Instruct"}}
        path = get_test_artifacts_path(params, base_dir="/tmp/exports")
        assert path.startswith("/tmp/exports/")

    def test_timestamp_included(self):
        params = {"model": {"model_id": "org/model"}}
        path = get_test_artifacts_path(params)
        # Should contain a timestamp like 2026-03-18_143022
        assert re.search(r"\d{4}-\d{2}-\d{2}_\d{6}", path)

    def test_default_base_dir(self):
        params = {"model": {"model_id": "org/model"}}
        path = get_test_artifacts_path(params)
        assert path.startswith("GenAILab/artifacts/exports/")
