# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
)

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2DecoderLayer,
)

from transformers.models.mistral.modeling_mistral import (
    MistralModel,
    MistralDecoderLayer,
)
import copy
from aimet_onnx.experimental.adascale.model_converter import ModelConverter

from dataclasses import dataclass
from typing import Type


# TODO Move AdaScaleModelConfig, adascale_model_config_dict to a utility file
@dataclass
class AdaScaleModelConfig:
    block_type: Type = None  # block types to use in a given model
    beta_gamma_lr: float = 1e-3  # lr for beta and gamma
    scales_lr: float = 5e-4  # lr for s2, s3, [s4]
    model_config: Type = None


# mapping of model type and the corresponding adascale config
adascale_model_config_dict = {
    "LlamaModel": AdaScaleModelConfig(
        block_type=LlamaDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "Qwen2Model": AdaScaleModelConfig(
        block_type=Qwen2DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "MistralModel": AdaScaleModelConfig(
        block_type=MistralDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
}


def test_get_decoder_blocks(monkeypatch):
    path = os.path.abspath(os.path.join("../../../../GenAITests"))
    monkeypatch.syspath_prepend(path)
    from GenAITests.onnx.models.qwen import Qwen_25_ONNX

    sim, config = Qwen_25_ONNX.instantiate_quantsim(
        "Qwen/Qwen2.5-1.5B", 4096, 2048, small_model=True
    )
    adascale_model_config_dict["Qwen2Model"].model_config = config

    converter = ModelConverter(sim, adascale_model_config_dict["Qwen2Model"])
    orig_random_pt_model = copy.deepcopy(converter.pt_decoder_blocks)
    for idx in range(len(orig_random_pt_model)):
        converter._copy_weights_onnx_to_pt(idx)

    for idx in range(2):
        assert torch.any(
            orig_random_pt_model[idx].self_attn.q_proj.weight
            != converter.get_pt_decoder_block(idx).self_attn.q_proj.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].self_attn.k_proj.weight
            != converter.get_pt_decoder_block(idx).self_attn.k_proj.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].self_attn.v_proj.weight
            != converter.get_pt_decoder_block(idx).self_attn.v_proj.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].self_attn.q_proj.bias
            != converter.get_pt_decoder_block(idx).self_attn.q_proj.bias
        )
        assert torch.any(
            orig_random_pt_model[idx].self_attn.k_proj.bias
            != converter.get_pt_decoder_block(idx).self_attn.k_proj.bias
        )
        assert torch.any(
            orig_random_pt_model[idx].self_attn.v_proj.bias
            != converter.get_pt_decoder_block(idx).self_attn.v_proj.bias
        )
        assert torch.any(
            orig_random_pt_model[idx].self_attn.o_proj.weight
            != converter.get_pt_decoder_block(idx).self_attn.o_proj.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].mlp.gate_proj.weight
            != converter.get_pt_decoder_block(idx).mlp.gate_proj.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].mlp.up_proj.weight
            != converter.get_pt_decoder_block(idx).mlp.up_proj.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].mlp.down_proj.weight
            != converter.get_pt_decoder_block(idx).mlp.down_proj.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].input_layernorm.weight
            != converter.get_pt_decoder_block(idx).input_layernorm.weight
        )
        assert torch.any(
            orig_random_pt_model[idx].post_attention_layernorm.weight
            != converter.get_pt_decoder_block(idx).post_attention_layernorm.weight
        )

    # Check if the params data in onnx initilizer's list == pytorch decoder blocks
    assert torch.all(
        converter.get_pt_decoder_block(0).self_attn.q_proj.weight
        == converter.initializer_map["onnx::MatMul_571"]
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).self_attn.q_proj.weight
        == converter.initializer_map["onnx::MatMul_647"]
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).self_attn.k_proj.weight
        == converter.initializer_map["onnx::MatMul_587"].T
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).self_attn.k_proj.weight
        == converter.initializer_map["onnx::MatMul_663"].T
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).self_attn.v_proj.weight
        == converter.initializer_map["onnx::MatMul_588"].T
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).self_attn.v_proj.weight
        == converter.initializer_map["onnx::MatMul_664"].T
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).self_attn.q_proj.bias
        == converter.initializer_map["model.model.layers.0.self_attn.q_proj.bias"]
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).self_attn.q_proj.bias
        == converter.initializer_map["model.model.layers.1.self_attn.q_proj.bias"]
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).self_attn.k_proj.bias
        == converter.initializer_map["model.model.layers.0.self_attn.k_proj.bias"]
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).self_attn.k_proj.bias
        == converter.initializer_map["model.model.layers.1.self_attn.k_proj.bias"]
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).self_attn.v_proj.bias
        == converter.initializer_map["model.model.layers.0.self_attn.v_proj.bias"]
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).self_attn.v_proj.bias
        == converter.initializer_map["model.model.layers.1.self_attn.v_proj.bias"]
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).self_attn.o_proj.weight
        == converter.initializer_map["onnx::MatMul_643"]
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).self_attn.o_proj.weight
        == converter.initializer_map["onnx::MatMul_719"]
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).mlp.gate_proj.weight
        == converter.initializer_map["onnx::MatMul_644"].T
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).mlp.gate_proj.weight
        == converter.initializer_map["onnx::MatMul_720"].T
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).mlp.up_proj.weight
        == converter.initializer_map["onnx::MatMul_645"].T
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).mlp.up_proj.weight
        == converter.initializer_map["onnx::MatMul_721"].T
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).mlp.down_proj.weight
        == converter.initializer_map["onnx::MatMul_646"].T
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).mlp.down_proj.weight
        == converter.initializer_map["onnx::MatMul_722"].T
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).input_layernorm.weight
        == converter.initializer_map["model.model.layers.0.input_layernorm.weight"]
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).input_layernorm.weight
        == converter.initializer_map["model.model.layers.1.input_layernorm.weight"]
    )

    assert torch.all(
        converter.get_pt_decoder_block(0).post_attention_layernorm.weight
        == converter.initializer_map[
            "model.model.layers.0.post_attention_layernorm.weight"
        ]
    )
    assert torch.all(
        converter.get_pt_decoder_block(1).post_attention_layernorm.weight
        == converter.initializer_map[
            "model.model.layers.1.post_attention_layernorm.weight"
        ]
    )
