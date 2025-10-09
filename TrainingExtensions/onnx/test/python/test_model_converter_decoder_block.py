# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from onnx import numpy_helper
import numpy as np

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
)

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
)

from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
)
from aimet_onnx.experimental.adascale.model_converter_decoder_block import (
    ModelConverter,
)

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


def test_decoder_block_weights_copy(monkeypatch, small_model=True):
    path = os.path.abspath(os.path.join("../../../../GenAITests"))
    monkeypatch.syspath_prepend(path)
    from transformers import AutoConfig
    from GenAITests.onnx.models.qwen import Qwen_25_ONNX

    model_id = "Qwen/Qwen2.5-0.5B"
    llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if small_model:
        llm_config.num_hidden_layers = 2

    sim = Qwen_25_ONNX.instantiate_quantsim(
        model_id, 4096, 2048, small_model=small_model
    )

    adascale_model_config_dict["Qwen2Model"].model_config = llm_config

    converter = ModelConverter(sim, adascale_model_config_dict["Qwen2Model"])

    pt_block = converter.get_pt_block(0)

    # Check if the params data in onnx initilizer's list == pytorch decoder blocks
    assert torch.all(
        pt_block.self_attn.q_proj.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map["onnx::MatMul_571"]
            ]
        ).T
    )

    assert torch.all(
        pt_block.self_attn.k_proj.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map["onnx::MatMul_587"]
            ]
        ).T
    )

    assert torch.all(
        pt_block.self_attn.v_proj.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map["onnx::MatMul_588"]
            ]
        ).T
    )

    assert torch.all(
        pt_block.self_attn.q_proj.bias
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map[
                    "model.model.layers.0.self_attn.q_proj.bias"
                ]
            ]
        )
    )

    assert torch.all(
        pt_block.self_attn.k_proj.bias
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map[
                    "model.model.layers.0.self_attn.k_proj.bias"
                ]
            ]
        )
    )

    assert torch.all(
        pt_block.self_attn.v_proj.bias
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map[
                    "model.model.layers.0.self_attn.v_proj.bias"
                ]
            ]
        )
    )

    assert torch.all(
        pt_block.self_attn.o_proj.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map["onnx::MatMul_643"]
            ]
        ).T
    )

    assert torch.all(
        pt_block.mlp.gate_proj.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map["onnx::MatMul_644"]
            ]
        ).T
    )

    assert torch.all(
        pt_block.mlp.up_proj.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map["onnx::MatMul_645"]
            ]
        ).T
    )

    assert torch.all(
        pt_block.mlp.down_proj.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map["onnx::MatMul_646"]
            ]
        ).T
    )

    assert torch.all(
        pt_block.input_layernorm.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map[
                    "model.model.layers.0.input_layernorm.weight"
                ]
            ]
        )
    )

    assert torch.all(
        pt_block.post_attention_layernorm.weight
        == numpy_helper.to_array(
            sim.model.model.graph.initializer[
                converter.initializer_name_to_index_map[
                    "model.model.layers.0.post_attention_layernorm.weight"
                ]
            ]
        )
    )


def test_model_round_trip(monkeypatch, small_model=True):
    path = os.path.abspath(os.path.join("../../../../GenAITests"))
    monkeypatch.syspath_prepend(path)
    from transformers import AutoConfig
    from GenAITests.onnx.models.qwen import Qwen_25_ONNX

    model_id = "Qwen/Qwen2.5-0.5B"
    sim = Qwen_25_ONNX.instantiate_quantsim(
        model_id, 4096, 2048, small_model=small_model
    )
    llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if small_model:
        llm_config.num_hidden_layers = 2
    adascale_model_config_dict["Qwen2Model"].model_config = llm_config

    def _update_onnx_weights(model, set_zeros: bool = False):
        for initializer in model.graph.initializer:
            weight_array = numpy_helper.to_array(initializer)
            new_array = (
                np.zeros_like(weight_array) if set_zeros else np.ones_like(weight_array)
            )
            new_initializer = numpy_helper.from_array(new_array, initializer.name)
            initializer.CopyFrom(new_initializer)

    def _check_onnx_weights(
        model, layers_to_check: set = None, are_zeros: bool = False
    ):
        for initializer in model.graph.initializer:
            if layers_to_check is not None and initializer.name not in layers_to_check:
                continue

            weight_array = numpy_helper.to_array(initializer)
            if are_zeros:
                assert (weight_array == 0.0).all()
            else:
                if not (weight_array == 1.0).all():
                    print(f"Weight mismatch {initializer.name}")
                else:
                    print(f"Weight Match {initializer.name}")
                assert (weight_array == 1.0).all()

    def _update_torch_weights(model, set_zeros: bool = False):
        for param in model.parameters():
            if set_zeros:
                param.data.zero_()
            else:
                param.data.fill_(1.0)

    def _check_torch_weights(model, are_zeros: bool = False):
        for param in model.parameters():
            if are_zeros:
                assert param.data.equal(torch.zeros_like(param.data))
            else:
                assert param.data.equal(torch.ones_like(param.data))

    # Update ONNX model weights to zeros
    _update_onnx_weights(sim.model.model, set_zeros=True)
    _check_onnx_weights(sim.model.model, are_zeros=True)

    converter = ModelConverter(sim, adascale_model_config_dict["Qwen2Model"])

    pt_block = converter.get_pt_block(0)
    # Check weights are zeros in pytorch decoder blocks
    _check_torch_weights(pt_block, are_zeros=True)
    # update pytorch weights to ones
    _update_torch_weights(pt_block, set_zeros=False)
    # Check weights are ones in pytorch decoder blocks
    _check_torch_weights(pt_block, are_zeros=False)
