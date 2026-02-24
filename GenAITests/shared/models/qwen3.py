# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-3 model class"""

import contextlib
import torch

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.qwen3 import modeling_qwen3

from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.adaptations import (
    replace_linears_with_convs,
    AdaptedModule,
)


class Qwen_3(LLM):
    """Generic quantized Qwen 2"""

    DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"

    @classmethod
    def instantiate_model(cls, model_id: str, small_model=False) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2
            if (
                hasattr(llm_config, "layer_types")
                and llm_config.layer_types is not None
            ):
                llm_config.layer_types = llm_config.layer_types[:2]
        return modeling_qwen3.Qwen3ForCausalLM.from_pretrained(
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


#################################  Extra code to enable SHA tests on Qwen 3  #################################


class MultiHeadLinear(torch.nn.Module):
    """Linear layer split into multiple heads that stitches outputs together"""

    def __init__(
        self, in_features: int, out_features: int, num_heads: int, bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features, self.head_dim, bias=bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each head and concatenate results"""
        # x shape: [batch_size, seq_len, in_features]
        head_outputs = [head(x) for head in self.heads]
        # Concatenate along the feature dimension
        return torch.cat(head_outputs, dim=-1)


class SHAQwen3Attention(modeling_qwen3.Qwen3Attention, AdaptedModule):
    """Split-Head Attention version of Qwen3Attention"""

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

    def adapt(self):
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
def enable_sha_qwen3_attention():
    original = modeling_qwen3.Qwen3Attention
    modeling_qwen3.Qwen3Attention = SHAQwen3Attention

    yield

    modeling_qwen3.Qwen3Attention = original


class Qwen_3_SHA_Mixin(Qwen_3):
    """Mixin class to convert instantiated qwen3 model to use split-head attention (SHA)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        with enable_sha_qwen3_attention():
            model = super().instantiate_model(*args, **kwargs)

        for module in model.modules():
            if isinstance(module, AdaptedModule):
                module.adapt()

        return model


#################################  Extra code to enable Conv tests on Qwen 3  #################################


class Qwen_3_SHA_Conv_Mixin(Qwen_3_SHA_Mixin):
    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        return replace_linears_with_convs(super().instantiate_model(*args, **kwargs))
