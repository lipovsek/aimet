# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Llama model class"""

import contextlib
import torch
from torch import nn

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.llama import modeling_llama

from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator


class Llama_32(LLM):
    """Generic LLaMa 3.2"""

    DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        )
        if small_model:
            llm_config.num_hidden_layers = 2
            if (
                hasattr(llm_config, "layer_types")
                and llm_config.layer_types is not None
            ):
                llm_config.layer_types = llm_config.layer_types[:2]
        return modeling_llama.LlamaForCausalLM.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> PreTrainedTokenizer:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def get_sample_backbone_inputs(cls, model, context_length, sequence_length):
        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = Generator.prepare_inputs(
            model=model,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
        )
        return assembled_dummy_inputs


#################################  Extra code to enable SHA tests on llama  #################################


class MultiHeadLinear(nn.Module):
    """Linear layer split into multiple heads that stitches outputs together"""

    def __init__(
        self, in_features: int, out_features: int, num_heads: int, bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.heads = nn.ModuleList(
            [nn.Linear(in_features, self.head_dim, bias=bias) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each head and concatenate results"""
        # x shape: [batch_size, seq_len, in_features]
        head_outputs = [head(x) for head in self.heads]
        # Concatenate along the feature dimension
        return torch.cat(head_outputs, dim=-1)


class SHALlamaAttention(modeling_llama.LlamaAttention):
    """Split-Head Attention version of LlamaAttention"""

    @property
    def hidden_size_(self):
        if hasattr(self, "hidden_size"):
            return self.hidden_size
        return self.config.hidden_size

    @property
    def num_attention_heads_(self):
        if hasattr(self, "num_heads"):
            return self.num_heads
        return self.config.num_attention_heads

    @property
    def num_key_value_heads_(self):
        if hasattr(self, "num_key_value_heads"):
            return self.num_key_value_heads
        return self.config.num_key_value_heads

    def apply_sha_adaptation(self):
        """Split q_proj, k_proj, and v_proj into separate linear layers per head"""

        # Create multi-head projections
        q_proj_mh = MultiHeadLinear(
            self.hidden_size_,
            self.num_attention_heads_ * self.head_dim,
            self.num_attention_heads_,
            bias=False,
        )
        k_proj_mh = MultiHeadLinear(
            self.hidden_size_,
            self.num_key_value_heads_ * self.head_dim,
            self.num_key_value_heads_,
            bias=False,
        )
        v_proj_mh = MultiHeadLinear(
            self.hidden_size_,
            self.num_key_value_heads_ * self.head_dim,
            self.num_key_value_heads_,
            bias=False,
        )

        # Copy weights from original projections
        for i in range(self.num_attention_heads_):
            q_proj_mh.heads[i].weight.data.copy_(
                self.q_proj.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )

        for i in range(self.num_key_value_heads_):
            k_proj_mh.heads[i].weight.data.copy_(
                self.k_proj.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )
            v_proj_mh.heads[i].weight.data.copy_(
                self.v_proj.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )

        # Replace original projections
        del self.q_proj
        del self.k_proj
        del self.v_proj

        self.q_proj = q_proj_mh
        self.k_proj = k_proj_mh
        self.v_proj = v_proj_mh


@contextlib.contextmanager
def enable_sha_llama_attention():
    if hasattr(modeling_llama, "LLAMA_ATTENTION_CLASSES"):
        original = modeling_llama.LLAMA_ATTENTION_CLASSES["eager"]
        modeling_llama.LLAMA_ATTENTION_CLASSES["eager"] = SHALlamaAttention
    else:
        original = modeling_llama.LlamaAttention
        modeling_llama.LlamaAttention = SHALlamaAttention

    yield

    if hasattr(modeling_llama, "LLAMA_ATTENTION_CLASSES"):
        modeling_llama.LLAMA_ATTENTION_CLASSES["eager"] = original
    else:
        modeling_llama.LlamaAttention = original


class Llama_32_SHA_Mixin(Llama_32):
    """Mixin class to convert instantiated llama model to use split-head attention (SHA)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        with enable_sha_llama_attention():
            model = Llama_32.instantiate_model(*args, **kwargs)

        for module in model.modules():
            if isinstance(module, SHALlamaAttention):
                module.apply_sha_adaptation()

        return model
