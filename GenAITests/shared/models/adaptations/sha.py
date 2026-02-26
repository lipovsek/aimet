# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Split-Head Attention (SHA) implementations for different model families.

SHA splits projection layers (q_proj, k_proj, v_proj) into separate linear
layers per head.
"""

import contextlib

import torch
from transformers import PreTrainedModel
from transformers.models.llama import modeling_llama
from transformers.models.qwen3 import modeling_qwen3

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.adaptations.base import AdaptedModule


# ============================================================================
# Multi-Head Linear (for SHA)
# ============================================================================


class MultiHeadLinear(torch.nn.Module):
    """Linear layer split into multiple heads that stitches outputs together.

    Used by SHA (Split-Head Attention) to split projection layers into
    per-head linear layers.
    """

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
        """Apply each head and concatenate results."""
        head_outputs = [head(x) for head in self.heads]
        return torch.cat(head_outputs, dim=-1)


# ============================================================================
# SHA (Split-Head Attention) Base Mixin
# ============================================================================


class SHAAttentionMixin(AdaptedModule):
    """Mixin providing SHA (Split-Head Attention) adaptation logic.

    Classes inheriting from this should also inherit from their
    model's attention class (e.g., LlamaAttention, Qwen3Attention).

    The attention class must have:
    - q_proj, k_proj, v_proj attributes (Linear layers)
    - head_dim attribute
    - Either hidden_size/num_heads/num_key_value_heads attributes
      or a config object with these values
    """

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
        """Split q_proj, k_proj, and v_proj into separate linear layers per head."""
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

        del self.q_proj
        del self.k_proj
        del self.v_proj

        self.q_proj = q_proj_mh
        self.k_proj = k_proj_mh
        self.v_proj = v_proj_mh


# ============================================================================
# Llama SHA Implementation
# ============================================================================


class SHALlamaAttention(SHAAttentionMixin, modeling_llama.LlamaAttention):
    """Split-Head Attention version of LlamaAttention."""

    pass


@contextlib.contextmanager
def enable_sha_llama_attention():
    """Context manager to temporarily replace LlamaAttention with SHA version."""
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


class LlamaSHAMixin:
    """Mixin to enable split-head attention (SHA) for Llama models."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        with enable_sha_llama_attention():
            model = super().instantiate_model(*args, **kwargs)

        for module in model.modules():
            if isinstance(module, AdaptedModule):
                module.adapt()

        return model


# ============================================================================
# Qwen3 SHA Implementation
# ============================================================================


class SHAQwen3Attention(SHAAttentionMixin, modeling_qwen3.Qwen3Attention):
    """Split-Head Attention version of Qwen3Attention."""

    pass


@contextlib.contextmanager
def enable_sha_qwen3_attention():
    """Context manager to temporarily replace Qwen3Attention with SHA version."""
    original = modeling_qwen3.Qwen3Attention
    modeling_qwen3.Qwen3Attention = SHAQwen3Attention

    yield

    modeling_qwen3.Qwen3Attention = original


class Qwen3SHAMixin:
    """Mixin to enable split-head attention (SHA) for Qwen3 models."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        with enable_sha_qwen3_attention():
            model = super().instantiate_model(*args, **kwargs)

        for module in model.modules():
            if isinstance(module, AdaptedModule):
                module.adapt()

        return model


# ============================================================================
# Registered SHA Adaptations
# ============================================================================


@YAMLConfigParser.register_adaptation("SHA", model_type="llama", exclusive=True)
class LlamaSHAAdaptation(LlamaSHAMixin):
    """SHA adaptation for Llama models."""

    pass


@YAMLConfigParser.register_adaptation("SHA", model_type="qwen3", exclusive=True)
class Qwen3SHAAdaptation(Qwen3SHAMixin):
    """SHA adaptation for Qwen3 models."""

    pass
