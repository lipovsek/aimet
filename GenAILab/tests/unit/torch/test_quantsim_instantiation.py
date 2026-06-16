# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Torch LLM quantsim instantiation logic.

These tests verify the orchestration in LLM_Torch.instantiate_quantsim
without loading real models — all heavy dependencies are mocked.
instantiate_quantsim now takes a pre-loaded float model (the runner loads
it via instantiate_float_model and may rotate it with SpinQuant first).
"""

from unittest.mock import MagicMock, patch, call

import pytest
import torch

from GenAILab.qai_hub_lm.precision import PrecisionConfig


@pytest.fixture
def _mock_torch_llm():
    """Patch heavy imports and return the LLM_Torch class ready for testing."""
    mock_quantsim_cls = MagicMock()
    mock_quantsim_instance = MagicMock()
    mock_quantsim_cls.return_value = mock_quantsim_instance

    # Create a module hierarchy that mimics a real model
    mock_rms_module = MagicMock()
    mock_linear_module = MagicMock()
    mock_lm_head = MagicMock()
    mock_quantsim_instance.model.modules.return_value = [
        mock_rms_module,
        mock_linear_module,
        mock_lm_head,
    ]
    mock_quantsim_instance.model.model.lm_head = mock_lm_head

    patches = {
        "quantsim_cls": mock_quantsim_cls,
        "quantsim": mock_quantsim_instance,
        "rms_module": mock_rms_module,
        "linear_module": mock_linear_module,
        "lm_head": mock_lm_head,
    }
    return patches


class TestInstantiateQuantsimOrchestration:
    """Test that instantiate_quantsim calls the right helpers in sequence."""

    def test_default_precision_used_when_none(self):
        """When precision=None, a default PrecisionConfig should be created."""
        with (
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QuantizationSimModel"
            ) as mock_qsim,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm._set_lm_head_precision"
            ) as mock_lm,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm._apply_block_granularity_to_decoder_stack"
            ) as mock_bg,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.ONNXExportableModuleWithCache"
            ) as mock_wrap,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.LLM_Torch.get_sample_backbone_inputs"
            ) as mock_sample,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QUANTSIM_CONFIG",
                "mock_config.json",
            ),
        ):
            mock_model = MagicMock()
            mock_wrap.return_value = mock_model

            mock_sim = MagicMock()
            mock_sim.model.modules.return_value = []
            mock_qsim.return_value = mock_sim

            from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

            result = LLM_Torch.instantiate_quantsim(
                mock_model,
                context_length=32,
                sequence_length=8,
                precision=None,
            )
            # _set_lm_head_precision should be called with default precision's lm_head
            mock_lm.assert_called_once()
            mock_bg.assert_called_once()

    def test_float16_activations_removes_quantizers(self):
        """When activations are float16, remove_activation_quantizers should be called."""
        from GenAILab.qai_hub_lm.precision import float16

        precision = PrecisionConfig.from_dict(
            {"blocks": {"default": {"qtype": 4}}, "activations": "float16"}
        )
        assert precision.activations == float16

        with (
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QuantizationSimModel"
            ) as mock_qsim,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.remove_activation_quantizers"
            ) as mock_remove,
            patch("GenAILab.qai_hub_lm.backends.torch.llm._set_lm_head_precision"),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm._apply_block_granularity_to_decoder_stack"
            ),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.ONNXExportableModuleWithCache"
            ) as mock_wrap,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.LLM_Torch.get_sample_backbone_inputs"
            ),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QUANTSIM_CONFIG",
                "mock_config.json",
            ),
        ):
            mock_model = MagicMock()
            mock_wrap.return_value = mock_model

            mock_sim = MagicMock()
            mock_sim.model.modules.return_value = []
            mock_qsim.return_value = mock_sim

            from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

            LLM_Torch.instantiate_quantsim(
                mock_model,
                context_length=32,
                sequence_length=8,
                precision=precision,
            )
            mock_remove.assert_called_once_with(mock_sim.model)

    def test_rms_norm_set_to_16_bits(self):
        """Quantized RMSNorm modules should have weight bitwidth set to 16."""
        with (
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QuantizationSimModel"
            ) as mock_qsim,
            patch("GenAILab.qai_hub_lm.backends.torch.llm._set_lm_head_precision"),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm._apply_block_granularity_to_decoder_stack"
            ),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.ONNXExportableModuleWithCache"
            ) as mock_wrap,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.LLM_Torch.get_sample_backbone_inputs"
            ),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QUANTSIM_CONFIG",
                "mock_config.json",
            ),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.LLM_Torch._is_quantized_rms_norm"
            ) as mock_is_rms,
        ):
            mock_model = MagicMock()
            mock_wrap.return_value = mock_model

            rms_module = MagicMock()
            rms_module.param_quantizers = {"weight": MagicMock(bitwidth=8)}
            non_rms_module = MagicMock()
            non_rms_module.param_quantizers = {"weight": MagicMock(bitwidth=8)}

            mock_sim = MagicMock()
            mock_sim.model.modules.return_value = [rms_module, non_rms_module]
            mock_qsim.return_value = mock_sim

            mock_is_rms.side_effect = lambda m: m is rms_module

            from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

            LLM_Torch.instantiate_quantsim(
                mock_model,
                context_length=32,
                sequence_length=8,
            )
            assert rms_module.param_quantizers["weight"].bitwidth == 16
            # non-RMS module should NOT have been set to 16
            assert non_rms_module.param_quantizers["weight"].bitwidth == 8

    def test_returns_sim_collection(self):
        """Should return a SimCollection with the quantsim as backbone."""
        with (
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QuantizationSimModel"
            ) as mock_qsim,
            patch("GenAILab.qai_hub_lm.backends.torch.llm._set_lm_head_precision"),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm._apply_block_granularity_to_decoder_stack"
            ),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.ONNXExportableModuleWithCache"
            ) as mock_wrap,
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.LLM_Torch.get_sample_backbone_inputs"
            ),
            patch(
                "GenAILab.qai_hub_lm.backends.torch.llm.QUANTSIM_CONFIG",
                "mock_config.json",
            ),
        ):
            mock_model = MagicMock()
            mock_wrap.return_value = mock_model

            mock_sim = MagicMock()
            mock_sim.model.modules.return_value = []
            mock_qsim.return_value = mock_sim

            from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch
            from GenAILab.qai_hub_lm.models.base import SimCollection

            result = LLM_Torch.instantiate_quantsim(
                mock_model,
                context_length=32,
                sequence_length=8,
            )
            assert isinstance(result, SimCollection)
            assert result.backbone is mock_sim


class TestIsQuantizedRmsNorm:
    def test_true_for_rms_norm(self):
        from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch
        from aimet_torch.v2.nn.true_quant import QuantizationMixin

        mock_module = MagicMock(spec=QuantizationMixin)
        mock_module.__class__ = type("MockRMS", (QuantizationMixin,), {})

        with patch(
            "GenAILab.qai_hub_lm.backends.torch.llm.map_torch_types_to_onnx",
            {type(mock_module): ["RMSNormalization"]},
        ):
            assert LLM_Torch._is_quantized_rms_norm(mock_module)

    def test_false_for_non_rms(self):
        from GenAILab.qai_hub_lm.backends.torch.llm import LLM_Torch

        mock_module = MagicMock()
        # Not a QuantizationMixin instance
        assert not LLM_Torch._is_quantized_rms_norm(mock_module)
