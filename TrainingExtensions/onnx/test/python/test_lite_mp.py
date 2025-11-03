# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import tempfile
import onnxruntime

from aimet_common.defs import QuantizationDataType
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx import analyze_per_layer_sensitivity
from aimet_onnx import int8, int16, float16
from aimet_onnx.utils import make_dummy_input, make_psnr_eval_fn
from aimet_onnx.lite_mp import flip_layers_to_higher_precision
from .models import models_for_tests


class TestLiteMp:
    @pytest.mark.parametrize("percent_flip", [30, 50.2, 80])
    def test_flip_to_float(self, percent_flip):
        model = models_for_tests.single_residual_model().model
        fp_session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )

        sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
        q_ops = [op for op in sim.qc_quantize_op_dict.values() if op.enabled]
        int8_count = sum(1 for op in q_ops if op.bitwidth == 8)
        fp_count = sum(1 for op in q_ops if op.data_type == QuantizationDataType.float)
        assert fp_count == 0

        inputs = [make_dummy_input(model)]
        psnr_eval_fn = make_psnr_eval_fn(fp_session, inputs)
        layer_sensitivity_dict = analyze_per_layer_sensitivity(
            sim, eval_fn=psnr_eval_fn
        )
        flip_layers_to_higher_precision(
            sim, layer_sensitivity_dict, percent_flip, override_precision=float16
        )

        new_int8_count = sum(1 for op in q_ops if op.bitwidth == 8 and op.enabled)
        assert new_int8_count < int8_count
        assert new_int8_count <= (100 - percent_flip) / 100 * int8_count

        sim.compute_encodings(inputs=[make_dummy_input(model)])

    @pytest.mark.parametrize("percent_flip", [30.5, 50, 80])
    def test_flip_to_int16(self, percent_flip):
        model = models_for_tests.single_residual_model().model
        fp_session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )

        sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
        q_ops = [op for op in sim.qc_quantize_op_dict.values() if op.enabled]
        int8_count = sum(1 for op in q_ops if op.bitwidth == 8)
        fp_count = sum(1 for op in q_ops if op.data_type == QuantizationDataType.float)
        assert fp_count == 0

        inputs = [make_dummy_input(model)]
        psnr_eval_fn = make_psnr_eval_fn(fp_session, inputs)
        layer_sensitivity_dict = analyze_per_layer_sensitivity(
            sim, eval_fn=psnr_eval_fn
        )
        flip_layers_to_higher_precision(
            sim, layer_sensitivity_dict, percent_flip, override_precision=int16
        )

        int16_count = sum(1 for op in q_ops if op.bitwidth == 16 and op.enabled)
        assert int16_count >= percent_flip / 100 * int8_count

        sim.compute_encodings(inputs=[make_dummy_input(model)])
