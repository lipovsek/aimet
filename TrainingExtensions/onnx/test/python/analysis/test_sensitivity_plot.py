# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for aimet_onnx.analysis.sensitivity_plot (HTML plot + JSON round-trip)."""

import os.path
import tempfile

import pytest

from aimet_onnx.analysis import (
    SensitivityMetric,
    save_sensitivity_plot,
    save_sensitivity_results,
    load_sensitivity_results,
)


def _metric(higher_is_worse=False):
    return SensitivityMetric("psnr", lambda s: 0.0, higher_is_worse=higher_is_worse)


# A small unordered score dict (ranking is applied by the API under test).
_SCORES = {
    "model.layers.0.q_proj.weight": 42.0,
    "model.layers.1.down_proj.weight": 30.0,
    "model.layers.2.k_proj.weight": 55.0,
}


class TestSaveSensitivityPlot:
    def test_writes_html(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sens.html")
            save_sensitivity_plot(_SCORES, _metric(), save_path=path)
            assert os.path.isfile(path)

    def test_custom_highlight_patterns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sens.html")
            save_sensitivity_plot(
                _SCORES, _metric(), save_path=path, highlight_patterns=["q_proj"]
            )
            assert os.path.isfile(path)

    def test_details_add_column_and_tooltip(self):
        details = {"model.layers.0.q_proj.weight": "onnx::MatMul_9772"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sens.html")
            save_sensitivity_plot(
                _SCORES,
                _metric(),
                save_path=path,
                details=details,
                details_label="Quantizer",
            )
            html = open(path, encoding="utf-8").read()
            assert "Quantizer" in html
            assert "onnx::MatMul_9772" in html

    def test_details_omitted_leaves_plot_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            without = os.path.join(tmp_dir, "without.html")
            save_sensitivity_plot(_SCORES, _metric(), save_path=without)
            assert "@detail" not in open(without, encoding="utf-8").read()

    def test_empty_scores_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError):
                save_sensitivity_plot(
                    {}, _metric(), save_path=os.path.join(tmp_dir, "s.html")
                )

    def test_non_html_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="'save_path' must end with '.html'."):
                save_sensitivity_plot(
                    _SCORES, _metric(), save_path=os.path.join(tmp_dir, "s.json")
                )

    def test_bad_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = tmp_dir
        with pytest.raises(NotADirectoryError):
            save_sensitivity_plot(
                _SCORES, _metric(), save_path=os.path.join(missing, "s.html")
            )


class TestSensitivityResultsRoundTrip:
    def test_json_round_trip_preserves_scores(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sens.json")
            save_sensitivity_results(_SCORES, save_path=path)
            loaded = load_sensitivity_results(path)
            assert loaded == _SCORES

    def test_json_carries_details(self):
        import json

        details = {"model.layers.1.down_proj.weight": "onnx::MatMul_9773"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sens.json")
            save_sensitivity_results(_SCORES, save_path=path, details=details)
            entries = json.load(open(path, encoding="utf-8"))["ranking"]
            by_name = {entry["name"]: entry for entry in entries}
            assert by_name["model.layers.1.down_proj.weight"]["detail"] == (
                "onnx::MatMul_9773"
            )
            # Names with no detail are written without the key.
            assert "detail" not in by_name["model.layers.0.q_proj.weight"]
            # Scores still round-trip unchanged.
            assert load_sensitivity_results(path) == _SCORES

    def test_json_preserves_input_order(self):
        # save_sensitivity_results preserves the (already-ranked) input order;
        # ranking is the analysis function's responsibility, not this writer's.
        ranked = {"b": 30.0, "a": 42.0, "c": 55.0}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sens.json")
            save_sensitivity_results(ranked, save_path=path)
            loaded = load_sensitivity_results(path)
            assert list(loaded.items()) == list(ranked.items())
