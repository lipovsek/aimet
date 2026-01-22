# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Test Omniquant functions. (Not include LET modules.)"""

import contextlib
import copy
import numpy as np
import os
from safetensors.numpy import save_file, load_file
import tempfile
import torch
from pathlib import Path
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.tuners.lora.layer import Linear as LoraLinear
import pytest
from aimet_torch.experimental.omniquant.defs import _LetPair

from aimet_torch.experimental.omniquant import decoder_processor
from aimet_torch.experimental.omniquant import omniquant_optimizer
from aimet_torch.experimental.omniquant._utils import (
    replace_with_omniquant_weight_quantizers,
    SUPPORTED_QUANTIZED_MODULES,
)
from aimet_torch.v2.quantsim import QuantizationSimModel


@contextlib.contextmanager
def add_custom_model_class_to_support_model_group(model_class, target_group_name):
    """
    Add model_class Class to decoder_processor.target_group_name.
    """
    target_group = getattr(decoder_processor, target_group_name)
    new_target_group = list(target_group)
    new_target_group.append(model_class)
    setattr(decoder_processor, target_group_name, tuple(new_target_group))

    yield

    setattr(decoder_processor, target_group_name, target_group)


@contextlib.contextmanager
def add_model_class_to_support_model_group_in_omniquant(model_class, target_group_name):
    """
    Add model_class Class to omniquant_optimizer.target_group_name.
    """
    target_group = getattr(omniquant_optimizer, target_group_name)
    new_target_group = list(target_group)
    new_target_group.append(model_class)
    setattr(omniquant_optimizer, target_group_name, tuple(new_target_group))

    yield

    setattr(decoder_processor, target_group_name, target_group)


@contextlib.contextmanager
def add_custom_model_block_to_support_block_type(target_support_block_map):
    block_map = getattr(decoder_processor, target_support_block_map)
    new_map = block_map
    new_map.update({FakeLlamaModel: FakeDecoderBlock})
    setattr(decoder_processor, target_support_block_map, new_map)

    yield

    setattr(decoder_processor, target_support_block_map, block_map)


def get_let_module_pair(decoder_block):
    """Method to get a list of let module pairs in a FakeDecoderBlock."""
    input_layernorm = decoder_block.get_submodule("input_layernorm")
    q_proj = decoder_block.get_submodule("self_attn.q_proj")
    k_proj = decoder_block.get_submodule("self_attn.k_proj")
    v_proj = decoder_block.get_submodule("self_attn.v_proj")
    o_proj = decoder_block.get_submodule("self_attn.o_proj")
    gate_proj = decoder_block.get_submodule("mlp.gate_proj")
    up_proj = decoder_block.get_submodule("mlp.up_proj")
    down_proj = decoder_block.get_submodule("mlp.down_proj")
    output_layernorm = decoder_block.get_submodule("post_attention_layernorm")
    return [
        _LetPair([input_layernorm], [q_proj, k_proj, v_proj]),
        _LetPair([v_proj], [o_proj]),
        _LetPair([output_layernorm], [gate_proj, up_proj]),
        _LetPair([up_proj], [down_proj]),
    ]


class FakeLlamaModel(torch.nn.Module):
    """Toy model for test"""

    def __init__(self, layer_num, seq_len, head_num, emb_dim):
        super().__init__()
        assert emb_dim % head_num == 0, "emb_dim need to be dividable by head_num."
        self.layers = torch.nn.ModuleList(
            [FakeDecoderBlock(seq_len, head_num, emb_dim) for _ in range(layer_num)]
        )
        self.out_linear = torch.nn.Linear(emb_dim, 5)

    def forward(self, x):
        """model forward"""
        for layer in self.layers:
            x = layer(x)
        x = self.out_linear(x)
        return x


class FakeDecoderBlock(torch.nn.Module):
    """Toy model for test"""

    def __init__(self, seq_len, head_num, emb_dim):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(emb_dim)
        self.self_attn = FakeSelfAttn(seq_len, head_num, emb_dim)
        self.mlp = FakeMlp(emb_dim)
        self.post_attention_layernorm = torch.nn.LayerNorm(emb_dim)

    def forward(self, x):
        """model forward"""
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.mlp(x)
        x = self.post_attention_layernorm(x)
        return x


class FakeSelfAttn(torch.nn.Module):
    """Toy model for test"""

    def __init__(self, seq_len, head_num, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.head_num = head_num
        self.emb_dim = emb_dim
        self.q_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.k_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.v_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.o_proj = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        """model forward"""
        head_dim = self.emb_dim // self.head_num
        q = (
            self.q_proj(x)
            .reshape(-1, self.seq_len, self.head_num, head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(-1, self.seq_len, self.head_num, head_dim)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v_proj(x)
            .reshape(-1, self.seq_len, self.head_num, head_dim)
            .permute(0, 2, 1, 3)
        )
        qk = torch.matmul(q, k)
        qkv = (
            torch.matmul(qk, v)
            .permute(0, 2, 1, 3)
            .reshape(-1, self.seq_len, self.head_num * head_dim)
        )
        out = self.o_proj(qkv)

        return out


class FakeMlp(torch.nn.Module):
    """Toy model for test"""

    def __init__(self, emb_dim):
        super().__init__()
        self.gate_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.up_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.down_proj = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        """model forward"""
        y0 = self.gate_proj(x)
        y1 = self.up_proj(x)
        y1 = self.down_proj(y1)
        return y0 + y1


class TestOmniquant:
    """Test Omniquant."""

    def test_get_transformer_processor(self):
        """Test get_transformer_processor returns correct TransformerProcessor and raise error."""
        layer_num = 5
        seq_len = 20
        head_num = 5
        emb_dim = 10
        dummy_input = torch.randn(1, seq_len, emb_dim)
        fake_llama_model = FakeLlamaModel(layer_num, seq_len, head_num, emb_dim)
        qsim = QuantizationSimModel(fake_llama_model, dummy_input)
        with torch.no_grad():
            # Make sure model is runnable.
            fake_llama_model(dummy_input)

        with add_custom_model_class_to_support_model_group(
            FakeLlamaModel, "LlamaModelGroup"
        ):
            with add_custom_model_block_to_support_block_type("model_to_block_mapping"):
                llama_processor = decoder_processor.get_transformer_processor(qsim)

                decoder_list = llama_processor.get_decoder_list(qsim)
                assert len(decoder_list) == layer_num

                for _decoder_block in decoder_list:
                    let_module_pair = get_let_module_pair(_decoder_block)
                    assert (
                        len(let_module_pair) == 4
                    )  # Llama Model Group should have 4 let pairs.

        with pytest.raises(ValueError):
            decoder_processor.get_transformer_processor(qsim)

    # pylint: disable=too-many-locals
    def test_dump_meta_data(self):
        """Test _dump_meta_data saves data and they are same as LET scale/"""
        layer_num = 5
        seq_len = 20
        head_num = 5
        emb_dim = 10
        dummy_input = torch.randn(1, seq_len, emb_dim)
        fake_llama_model = FakeLlamaModel(layer_num, seq_len, head_num, emb_dim)
        qsim = QuantizationSimModel(fake_llama_model, dummy_input)
        omniquant = omniquant_optimizer.Omniquant()

        with add_custom_model_class_to_support_model_group(
            FakeLlamaModel, "LlamaModelGroup"
        ):
            with add_custom_model_block_to_support_block_type("model_to_block_mapping"):
                with torch.no_grad():
                    llama_processor = decoder_processor.get_transformer_processor(qsim)
                    decoder_list = llama_processor.get_decoder_list(qsim)
                    replace_with_omniquant_weight_quantizers(decoder_list)
                    for _decoder_block in decoder_list:
                        qt_let_pair_list = get_let_module_pair(_decoder_block)
                        llama_processor.init_let_params(qt_let_pair_list, num_repeats=1)

                        # Manual apply scale to Let Module
                        for let_pair in qt_let_pair_list:
                            prev, foll = let_pair.prev, let_pair.follow
                            scale = torch.randn(emb_dim)
                            quantizer = prev[0].param_quantizers["weight"]
                            quantizer.prev_scale = torch.nn.Parameter(
                                quantizer.prev_scale * scale
                            )
                            for _foll in foll:
                                quantizer = _foll.param_quantizers["weight"]
                                quantizer.foll_scale = torch.nn.Parameter(
                                    quantizer.foll_scale * scale
                                )

                        for module in _decoder_block.modules():
                            if isinstance(module, SUPPORTED_QUANTIZED_MODULES):
                                for key, quantizer in module.param_quantizers.items():
                                    quantizer.fold_let_params(module, key)

                    # pylint: disable=protected-access
                    with tempfile.TemporaryDirectory() as tempdir:
                        omniquant._dump_meta_data(qsim.model, Path(tempdir))
                        metadata_path = os.path.join(
                            tempdir, "aimet_omniquant_metadata.safetensor"
                        )
                        metadata = load_file(metadata_path)
                        assert len(metadata) == 55  # There should be 55 scales dumped.

                    for k, metadata_scale in metadata.items():
                        module_name = ".".join(k.split(".")[:-1])
                        prev_foll = k.split(".")[-1]
                        let_module = qsim.model.get_submodule(module_name)
                        quantizer = let_module.param_quantizers["weight"]
                        cached_scale = getattr(
                            quantizer, "_cached_" + prev_foll + "_scale"
                        )

                        # cached_scale and metadata_scale is numpy array
                        assert (np.equal(cached_scale, metadata_scale)).all()

    # pylint: disable=too-many-locals
    def test_load_lora_model(self):
        """Test omniquant_optimizer.update_lora_weights"""
        layer_num = 2
        seq_len = 20
        head_num = 5
        emb_dim = 10
        dummy_input = torch.randn(1, seq_len, emb_dim)
        let_model = FakeLlamaModel(layer_num, seq_len, head_num, emb_dim)
        ori_model = copy.deepcopy(let_model)

        input_layernorm_scale = torch.nn.Parameter(torch.randn(emb_dim))
        mlp_scale = torch.nn.Parameter(torch.randn(emb_dim))
        input_layernorm_scale_2 = torch.nn.Parameter(torch.randn(emb_dim))
        mlp_scale_2 = torch.nn.Parameter(torch.randn(emb_dim))
        meta_data = {
            "layers.0.input_layernorm.prev": input_layernorm_scale,
            "layers.0.self_attn.q_proj.foll": input_layernorm_scale,
            "layers.0.self_attn.k_proj.foll": input_layernorm_scale,
            "layers.0.self_attn.v_proj.foll": input_layernorm_scale,
            "layers.0.mlp.up_proj.prev": mlp_scale,
            "layers.0.mlp.down_proj.foll": mlp_scale,
            "layers.1.input_layernorm.prev": input_layernorm_scale_2,
            "layers.1.self_attn.q_proj.foll": input_layernorm_scale_2,
            "layers.1.self_attn.k_proj.foll": input_layernorm_scale_2,
            "layers.1.self_attn.v_proj.foll": input_layernorm_scale_2,
            "layers.1.mlp.up_proj.prev": mlp_scale_2,
            "layers.1.mlp.down_proj.foll": mlp_scale_2,
        }

        # Apply meta data to let_model's weight direrctly.
        with torch.no_grad():
            for layer_name, scale in meta_data.items():
                module = let_model.get_submodule(layer_name[:-5])
                if layer_name.endswith("prev"):
                    if module.bias is not None:
                        module.bias.copy_(module.bias / scale)
                    new_weight = module.weight / (
                        scale.reshape(-1, 1)
                        if isinstance(module, torch.nn.Linear)
                        else scale
                    )
                    module.weight.copy_(new_weight)

                elif layer_name.endswith("foll"):
                    module.weight.copy_(module.weight * scale.reshape(1, -1))

        let_output = let_model(dummy_input)
        ori_output = ori_model(dummy_input)
        assert torch.allclose(let_output, ori_output, atol=1e-5)

        # Set model to lora model
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=2,
            bias="none",
            target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        )
        peft_ori_model = get_peft_model(ori_model, lora_config)
        peft_let_model = get_peft_model(let_model, lora_config)
        peft_ori_model.eval()  # .eval() to disable lora dropout.
        peft_let_model.eval()  # .eval() to disable lora dropout.

        # Lora B default init to zero weight.
        # Init lora B weight to random weight.
        with torch.no_grad():
            for _, module in peft_ori_model.named_modules():
                if isinstance(module, LoraLinear):
                    for _, _lora_b in module.lora_B.items():
                        _lora_b.weight = torch.nn.Parameter(
                            torch.randn(_lora_b.weight.shape)
                        )
        # Copy peft weight to let peft model.
        set_peft_model_state_dict(
            peft_let_model, get_peft_model_state_dict(peft_ori_model)
        )

        # Run omniquant_optimizer.update_lora_weights
        with tempfile.TemporaryDirectory() as tempdir:
            metadata_path = os.path.join(tempdir, "./metadata.safetensor")
            save_file({k: v.data.numpy() for k, v in meta_data.items()}, metadata_path)
            omniquant_optimizer.update_lora_weights(peft_let_model, metadata_path)

        peft_let_output = peft_let_model(dummy_input)
        peft_ori_output = peft_ori_model(dummy_input)

        # peft model output should be different from base model output.
        assert not torch.allclose(let_output, peft_let_output, atol=1e-5)
        assert torch.allclose(peft_let_output, peft_ori_output, atol=1e-5)


test = TestOmniquant()
test.test_dump_meta_data()
