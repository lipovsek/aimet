# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX quantsim utility functions."""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestGetOrtProviders:
    def test_cpu_device(self):
        from GenAILab.onnx.models.utils.quantsim_utils import get_ort_providers

        providers = get_ort_providers(torch.device("cpu"))
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_device_with_index(self):
        from GenAILab.onnx.models.utils.quantsim_utils import get_ort_providers

        providers = get_ort_providers(torch.device("cuda", 0))
        assert len(providers) == 2
        assert providers[0] == ("CUDAExecutionProvider", {"device_id": 0})
        assert providers[1] == "CPUExecutionProvider"

    def test_cuda_device_without_index(self):
        from GenAILab.onnx.models.utils.quantsim_utils import get_ort_providers

        providers = get_ort_providers(torch.device("cuda"))
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


class TestAttributePatch:
    def test_sets_and_restores_attribute(self):
        from GenAILab.onnx.models.utils.quantsim_utils import AttributePatch

        class Obj:
            x = 10

        obj = Obj()
        with AttributePatch(obj, "x", 42):
            assert obj.x == 42
        assert obj.x == 10

    def test_creates_and_removes_new_attribute(self):
        from GenAILab.onnx.models.utils.quantsim_utils import AttributePatch

        class Obj:
            pass

        obj = Obj()
        assert not hasattr(obj, "y")
        with AttributePatch(obj, "y", 99):
            assert obj.y == 99
        assert not hasattr(obj, "y")

    def test_restores_class_level_attribute(self):
        from GenAILab.onnx.models.utils.quantsim_utils import AttributePatch

        class Obj:
            z = "original"

        obj = Obj()
        with AttributePatch(obj, "z", "patched"):
            assert obj.z == "patched"
        # After exit, the instance override is removed, class attr remains
        assert obj.z == "original"


class TestTieQuantizersForKvCache:
    def test_ties_past_key_value_quantizers(self):
        from GenAILab.onnx.models.utils.quantsim_utils import (
            _tie_quantizers_for_kv_cache,
        )

        mock_input_0 = MagicMock()
        mock_input_0.name = "past_key_0_in"
        mock_input_1 = MagicMock()
        mock_input_1.name = "past_value_0_in"

        mock_graph = MagicMock()
        mock_graph.input = [mock_input_0, mock_input_1]

        mock_qsim = MagicMock()
        mock_qsim.model.graph.return_value = mock_graph

        out_quantizer_key = MagicMock()
        out_quantizer_val = MagicMock()
        mock_qsim.qc_quantize_op_dict = {
            "past_key_0_out": out_quantizer_key,
            "past_value_0_out": out_quantizer_val,
        }

        _tie_quantizers_for_kv_cache(mock_qsim)

        mock_qsim.set_quantizers.assert_called_once()
        mapping = mock_qsim.set_quantizers.call_args[0][0]
        assert mapping["past_key_0_in"] is out_quantizer_key
        assert mapping["past_value_0_in"] is out_quantizer_val


class TestRemoveActivationQuantizers:
    def test_disables_activation_quantizers(self):
        from GenAILab.onnx.models.utils.quantsim_utils import (
            _remove_activation_quantizers,
        )

        mock_qsim = MagicMock()
        act_op = MagicMock()
        weight_op = MagicMock()
        mock_qsim.qc_quantize_op_dict = {
            "act_tensor": act_op,
            "weight_tensor": weight_op,
        }
        mock_qsim.activation_names = {"act_tensor"}

        _remove_activation_quantizers(mock_qsim)

        act_op.reset_encoding_stats.assert_called_once()
        assert act_op.enabled is False
        weight_op.reset_encoding_stats.assert_not_called()


class TestResolveKvCacheQuantization:
    def test_float_precision_skips_tying(self):
        from GenAILab.onnx.models.utils.quantsim_utils import (
            _resolve_kv_cache_quantization,
        )
        from aimet_onnx.common.defs import float16

        mock_qsim = MagicMock()
        # Should not raise or call tie
        _resolve_kv_cache_quantization(mock_qsim, float16)
        mock_qsim.set_quantizers.assert_not_called()

    def test_int_precision_ties_and_sets_symmetric(self):
        from GenAILab.onnx.models.utils.quantsim_utils import (
            _resolve_kv_cache_quantization,
            _tie_quantizers_for_kv_cache,
        )
        from aimet_onnx.common.defs import qtype

        mock_qsim = MagicMock()
        mock_graph = MagicMock()
        mock_graph.input = []
        mock_graph.output = []
        mock_qsim.model.graph.return_value = mock_graph

        int8 = qtype.int(8)
        _resolve_kv_cache_quantization(mock_qsim, int8)
        # Should have called set_quantizers (from _tie_quantizers_for_kv_cache)
        mock_qsim.set_quantizers.assert_called()
