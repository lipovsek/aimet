# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for profiler stats writing and merging."""

import csv
import json
import os

import pytest

from GenAILab.shared.helpers.profiler import (
    write_stats_to_disk,
    merge_json_results,
    merge_csv_results,
    convert_gpu_meter_to_dict,
    _collect_environment,
    ComponentRecipeStats,
    RecipeStepStats,
    MetricResult,
)


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return str(d)


def _write_sample(results_dir, model_type="llama", model_id="org/model"):
    components = {
        "backbone": ComponentRecipeStats(
            steps=[
                RecipeStepStats(
                    recipe_name="Calibration",
                    recipe_kwargs={"num_batches": 32},
                    dataset_name="C4",
                    dataset_kwargs={"split": "en"},
                    profiler=None,
                )
            ]
        )
    }
    accuracy_results = [
        MetricResult(metric_name="PPL", result=12.5, profiler=None),
    ]
    write_stats_to_disk(
        output_folder=results_dir,
        filename="profiling_data",
        model_type=model_type,
        model_id=model_id,
        model_modifiers={"context_length": 64},
        components=components,
        accuracy_results=accuracy_results,
    )


class TestWriteStats:
    def test_json_creates_file(self, results_dir):
        _write_sample(results_dir)
        json_path = os.path.join(results_dir, "profiling_data.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert "llama" in data
        assert len(data["llama"]) == 1
        assert data["llama"][0]["model_id"] == "org/model"

    def test_json_appends(self, results_dir):
        _write_sample(results_dir, model_id="org/model1")
        _write_sample(results_dir, model_id="org/model2")
        json_path = os.path.join(results_dir, "profiling_data.json")
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["llama"]) == 2

    def test_csv_creates_with_header(self, results_dir):
        _write_sample(results_dir)
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == [
            "model_type",
            "model_id",
            "model_modifiers",
            "precision",
            "components",
            "accuracy_results",
            "export",
            "environment",
            "run_group",
        ]
        assert len(rows) == 2  # header + 1 data row

    def test_csv_appends_without_extra_header(self, results_dir):
        _write_sample(results_dir, model_id="org/model1")
        _write_sample(results_dir, model_id="org/model2")
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        # First row is header
        assert rows[0][0] == "model_type"

    def test_csv_json_fields_escaped(self, results_dir):
        _write_sample(results_dir)
        csv_path = os.path.join(results_dir, "profiling_data.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # model_modifiers column uses postgres CSV JSON format (quoted + escaped)
        raw = rows[1][2]
        # Unwrap postgres format: strip outer quotes, unescape doubled quotes
        inner = raw.strip('"').replace('""', '"')
        parsed = json.loads(inner)
        assert parsed["context_length"] == 64


class TestMerge:
    def test_merge_json(self, tmp_path):
        src = tmp_path / "src.json"
        dst = tmp_path / "dst.json"
        src.write_text(json.dumps({"llama": [{"model_id": "a"}]}))
        dst.write_text(json.dumps({"llama": [{"model_id": "b"}]}))
        count = merge_json_results(str(src), str(dst))
        assert count == 1
        with open(dst) as f:
            data = json.load(f)
        assert len(data["llama"]) == 2

    def test_merge_json_empty_source(self, tmp_path):
        count = merge_json_results(
            str(tmp_path / "nonexistent.json"), str(tmp_path / "dst.json")
        )
        assert count == 0

    def test_merge_csv(self, tmp_path):
        src = tmp_path / "src.csv"
        dst = tmp_path / "dst.csv"
        src.write_text("a,b\n1,2\n3,4\n")
        dst.write_text("a,b\n5,6\n")
        count = merge_csv_results(str(src), str(dst))
        assert count == 2
        with open(dst) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4  # header + 1 existing + 2 new

    def test_merge_csv_creates_dest(self, tmp_path):
        src = tmp_path / "src.csv"
        dst = tmp_path / "dst.csv"
        src.write_text("a,b\n1,2\n")
        count = merge_csv_results(str(src), str(dst))
        assert count == 1
        assert dst.exists()


class TestHelpers:
    def test_collect_environment(self):
        env = _collect_environment()
        assert "python_version" in env
        assert "platform" in env
        assert env["run_type"] == "local"

    def test_collect_environment_cached(self):
        env1 = _collect_environment()
        env2 = _collect_environment()
        assert env1 is env2

    def test_convert_gpu_meter_none(self):
        assert convert_gpu_meter_to_dict(None) == {}
