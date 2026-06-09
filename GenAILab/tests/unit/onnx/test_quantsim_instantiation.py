# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX LLM quantsim instantiation logic.

These tests verify the orchestration in LLM_ONNX.instantiate_quantsim
without loading real models — all heavy dependencies are mocked.
"""

from unittest.mock import MagicMock, patch, call

import pytest

from GenAILab.qai_hub_lm.precision import PrecisionConfig


class TestOnnxInstantiateQuantsimOrchestration:
    def _run_instantiate(self, precision=None, extra_patches=None):
        """Helper to run instantiate_quantsim with all heavy deps mocked."""
        patches = {
            "GenAILab.qai_hub_lm.backends.onnx.llm.is_huggingface_ckpt": MagicMock(
                return_value=False
            ),
            "GenAILab.qai_hub_lm.backends.onnx.llm.load_model_components_from_disk": MagicMock(
                return_value=(MagicMock(), None)
            ),
            "GenAILab.qai_hub_lm.backends.onnx.llm.AutoConfig": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.QuantizationSimModel": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm._set_lm_head_precision": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm._resolve_kv_cache_quantization": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm._apply_block_granularity_to_decoder_stack": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm._remove_activation_quantizers": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.get_ort_providers": MagicMock(
                return_value=["CPUExecutionProvider"]
            ),
            "GenAILab.qai_hub_lm.backends.onnx.llm.AttributePatch": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.QUANTSIM_CONFIG": "config.json",
        }
        if extra_patches:
            patches.update(extra_patches)

        active_patches = {k: patch(k, v) for k, v in patches.items()}
        mocks = {}
        for k, p in active_patches.items():
            mocks[k] = p.start()

        try:
            from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX

            entry = LLM_ONNX.export_onnx_models(
                model_id="/fake/path",
                context_length=32,
                sequence_length=8,
            )
            result = LLM_ONNX.instantiate_quantsim(entry, precision=precision)
            return result, mocks
        finally:
            for p in active_patches.values():
                p.stop()

    def test_default_precision_when_none(self):
        result, mocks = self._run_instantiate(precision=None)
        # Should not fail — default PrecisionConfig should be used
        from GenAILab.qai_hub_lm.models.base import SimCollection

        assert isinstance(result, SimCollection)

    def test_calls_set_lm_head_precision(self):
        result, mocks = self._run_instantiate()
        mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm._set_lm_head_precision"
        ].assert_called_once()

    def test_calls_resolve_kv_cache(self):
        result, mocks = self._run_instantiate()
        mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm._resolve_kv_cache_quantization"
        ].assert_called_once()

    def test_calls_block_granularity(self):
        result, mocks = self._run_instantiate()
        mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm._apply_block_granularity_to_decoder_stack"
        ].assert_called_once()

    def test_non_float32_does_not_remove_activations(self):
        precision = PrecisionConfig.from_dict(
            {"blocks": {"default": {"qtype": 4}}, "activations": 8}
        )
        result, mocks = self._run_instantiate(precision=precision)
        mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm._remove_activation_quantizers"
        ].assert_not_called()

    def test_returns_sim_collection_with_config(self):
        result, mocks = self._run_instantiate()
        from GenAILab.qai_hub_lm.models.base import SimCollection

        assert isinstance(result, SimCollection)
        # Config should be set from AutoConfig
        assert result.config is not None
