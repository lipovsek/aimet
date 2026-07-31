# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for aimet_onnx.analysis.sensitivity (GenAILab-free, tiny ONNX model)."""

import onnxruntime
import pytest

from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
from aimet_onnx import int8
from aimet_onnx.utils import make_dummy_input, make_psnr_eval_fn
from aimet_onnx.analysis import (
    SensitivityMetric,
    make_topk_logit_psnr_metric,
    analyze_per_quantizer_sensitivity,
)

from ..models import models_for_tests


def _calibrated_sim():
    """Build and calibrate a small W8A8 sim plus its FP session and inputs."""
    model = models_for_tests.single_residual_model().model
    fp_session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
    inputs = [make_dummy_input(model)]
    with compute_encodings(sim):
        sim.session.run(None, inputs[0])
    return sim, fp_session, inputs


class TestSensitivityMetric:
    def test_ranking_direction(self):
        higher = SensitivityMetric("h", lambda s: 0.0, higher_is_worse=True)
        lower = SensitivityMetric("l", lambda s: 0.0, higher_is_worse=False)
        assert higher.sensitivity_score(5.0) == 5.0
        assert lower.sensitivity_score(5.0) == -5.0

    def test_non_callable_eval_fn_raises(self):
        with pytest.raises(ValueError):
            SensitivityMetric("bad", eval_fn=object())


class TestPerQuantizerSensitivity:
    def test_returns_score_per_quantizer(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        scores = analyze_per_quantizer_sensitivity(sim, metric)

        enabled = [n for n, q in sim.qc_quantize_op_dict.items() if q.enabled]
        assert set(scores) == set(enabled)
        assert all(isinstance(v, float) for v in scores.values())

    def test_ordered_most_sensitive_first(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        scores = analyze_per_quantizer_sensitivity(sim, metric)
        ranked = [metric.sensitivity_score(v) for v in scores.values()]
        assert ranked == sorted(ranked, reverse=True)

    def test_group_fn_collapses_units(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        # Collapse all quantizers into one group -> a single score entry.
        scores = analyze_per_quantizer_sensitivity(
            sim, metric, group_fn=lambda name: "all"
        )
        assert list(scores) == ["all"]

    def test_group_fn_none_skips_quantizer(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        # Drop everything -> no groups -> error.
        with pytest.raises(RuntimeError):
            analyze_per_quantizer_sensitivity(sim, metric, group_fn=lambda name: None)

    def test_restores_enabled_state(self):
        sim, fp_session, inputs = _calibrated_sim()
        before = {n: q.enabled for n, q in sim.qc_quantize_op_dict.items()}
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        analyze_per_quantizer_sensitivity(sim, metric)
        after = {n: q.enabled for n, q in sim.qc_quantize_op_dict.items()}
        assert before == after

    def test_group_fn_subset_restricts_sweep(self):
        # A group_fn that returns None for all but a chosen subset (the pattern
        # used for KV-cache-only sweeps) restricts the analysis to that subset.
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        selected = [n for n, q in sim.qc_quantize_op_dict.items() if q.enabled][:2]
        scores = analyze_per_quantizer_sensitivity(
            sim, metric, group_fn=lambda name: name if name in selected else None
        )
        assert set(scores) == set(selected)


class TestTopkLogitPsnrMetric:
    def test_returns_float(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = make_topk_logit_psnr_metric(fp_session, inputs, k=5)
        assert metric.higher_is_worse is False
        score = metric(sim.session)
        assert isinstance(score, float)
