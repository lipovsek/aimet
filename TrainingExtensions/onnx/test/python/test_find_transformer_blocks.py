# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import sys
import platform
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
    get_conv_linear_layers_decoder_block,
)
from .utils import add_genai_tests_path
from .conftest import skip_module_on_windows_arm64

skip_module_on_windows_arm64(
    "transformers and onnx_sim is not available on Windows ARM64"
)


def verify_find_blocks(sim, model_type):
    end_points = get_decoder_blocks_end_points(sim, model_type)
    end_points_names = [(op1.name, op2.name) for op1, op2 in end_points]
    assert end_points_names == [
        (
            "/model/model/layers.0/input_layernorm/Pow",
            "/model/model/layers.1/input_layernorm/Pow",
        ),
        ("/model/model/layers.1/input_layernorm/Pow", "/model/model/norm/Pow"),
    ]
    conv_linear_blocks = get_conv_linear_layers_decoder_block(sim, end_points)
    conv_linear_blocks_names = []
    for ops in conv_linear_blocks:
        res = []
        for op in ops:
            res.append(op.name)
        conv_linear_blocks_names.append(res)

    assert conv_linear_blocks_names == [
        [
            "/model/model/layers.0/self_attn/v_proj/MatMul",
            "/model/model/layers.0/self_attn/k_proj/MatMul",
            "/model/model/layers.0/self_attn/q_proj/MatMul",
            "/model/model/layers.0/self_attn/MatMul",
            "/model/model/layers.0/self_attn/MatMul_1",
            "/model/model/layers.0/self_attn/o_proj/MatMul",
            "/model/model/layers.0/mlp/up_proj/MatMul",
            "/model/model/layers.0/mlp/gate_proj/MatMul",
            "/model/model/layers.0/mlp/down_proj/MatMul",
        ],
        [
            "/model/model/layers.1/self_attn/v_proj/MatMul",
            "/model/model/layers.1/self_attn/k_proj/MatMul",
            "/model/model/layers.1/self_attn/q_proj/MatMul",
            "/model/model/layers.1/self_attn/MatMul",
            "/model/model/layers.1/self_attn/MatMul_1",
            "/model/model/layers.1/self_attn/o_proj/MatMul",
            "/model/model/layers.1/mlp/up_proj/MatMul",
            "/model/model/layers.1/mlp/gate_proj/MatMul",
            "/model/model/layers.1/mlp/down_proj/MatMul",
        ],
    ]


def test_get_decoder_blocks(add_genai_tests_path):
    from GenAITests.onnx.models.qwen2 import Qwen_25_ONNX

    sim = Qwen_25_ONNX.instantiate_quantsim("Qwen/Qwen2-0.5B", 32, 16, small_model=True)
    verify_find_blocks(sim, "qwen2")


def test_get_decoder_blocks_qwen3(add_genai_tests_path):
    from GenAITests.onnx.models.qwen3 import Qwen_3_ONNX

    sim = Qwen_3_ONNX.instantiate_quantsim("Qwen/Qwen3-0.6B", 32, 16, small_model=True)
    verify_find_blocks(sim, "qwen3")


@pytest.mark.skip(reason="This takes long to run, similar test for Qwen exists")
def test_get_decoder_blocks_phi(add_genai_tests_path):
    from GenAITests.onnx.models.phi3 import Phi_3_ONNX

    sim = Phi_3_ONNX.instantiate_quantsim(
        "microsoft/Phi-3-mini-4k-instruct", 32, 16, small_model=True
    )
    end_points = get_decoder_blocks_end_points(sim, "phi3")
    end_points_names = [(op1.name, op2.name) for op1, op2 in end_points]
    assert end_points_names == [
        (
            "/model/model/layers.0/input_layernorm/Pow",
            "/model/model/layers.1/input_layernorm/Pow",
        ),
        ("/model/model/layers.1/input_layernorm/Pow", "/model/model/norm/Pow"),
    ]
    conv_linear_blocks = get_conv_linear_layers_decoder_block(sim, end_points)
    conv_linear_blocks_names = []
    for ops in conv_linear_blocks:
        res = []
        for op in ops:
            res.append(op.name)
        conv_linear_blocks_names.append(res)

    assert conv_linear_blocks_names == [
        [
            "/model/model/layers.0/self_attn/qkv_proj/MatMul",
            "/model/model/layers.0/self_attn/MatMul",
            "/model/model/layers.0/self_attn/MatMul_1",
            "/model/model/layers.0/self_attn/o_proj/MatMul",
            "/model/model/layers.0/mlp/gate_up_proj/MatMul",
            "/model/model/layers.0/mlp/down_proj/MatMul",
        ],
        [
            "/model/model/layers.1/self_attn/qkv_proj/MatMul",
            "/model/model/layers.1/self_attn/MatMul",
            "/model/model/layers.1/self_attn/MatMul_1",
            "/model/model/layers.1/self_attn/o_proj/MatMul",
            "/model/model/layers.1/mlp/gate_up_proj/MatMul",
            "/model/model/layers.1/mlp/down_proj/MatMul",
        ],
    ]
