# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Split Fused Layers (SplitFusedLayers) implementations.

Splits fused/joint projection layers into separate ones:
- qkv_proj -> q_proj, k_proj, v_proj  (attention)
- gate_up_proj -> gate_proj, up_proj   (MLP)

This targets models like Phi-3 that use fused projections.
"""

import contextlib
from collections.abc import Callable

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.phi3 import modeling_phi3
from transformers.models.phi3.modeling_phi3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.transforms.base import AdaptedModule


# ============================================================================
# Generic Mixins
# ============================================================================


class SplitQKVAttentionMixin(AdaptedModule):
    """Mixin that splits a fused qkv_proj into separate q_proj, k_proj, v_proj.

    The attention class must have:
    - qkv_proj attribute (nn.Linear with output = [Q|K|V] concatenated)
    - config with hidden_size, num_attention_heads, num_key_value_heads
    - head_dim attribute or derivable from config
    """

    def adapt(self):
        """Split qkv_proj into separate q_proj, k_proj, v_proj Linear layers."""
        config = self.config
        head_dim = getattr(
            self, "head_dim", config.hidden_size // config.num_attention_heads
        )
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        has_bias = self.qkv_proj.bias is not None

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        q_proj = torch.nn.Linear(hidden_size, q_size, bias=has_bias)
        k_proj = torch.nn.Linear(hidden_size, kv_size, bias=has_bias)
        v_proj = torch.nn.Linear(hidden_size, kv_size, bias=has_bias)

        q_proj.weight.data.copy_(self.qkv_proj.weight[:q_size, :])
        k_proj.weight.data.copy_(self.qkv_proj.weight[q_size : q_size + kv_size, :])
        v_proj.weight.data.copy_(self.qkv_proj.weight[q_size + kv_size :, :])

        if has_bias:
            q_proj.bias.data.copy_(self.qkv_proj.bias[:q_size])
            k_proj.bias.data.copy_(self.qkv_proj.bias[q_size : q_size + kv_size])
            v_proj.bias.data.copy_(self.qkv_proj.bias[q_size + kv_size :])

        del self.qkv_proj
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj


class SplitGateUpMLPMixin(AdaptedModule):
    """Mixin that splits a fused gate_up_proj into separate gate_proj, up_proj.

    The MLP class must have:
    - gate_up_proj attribute (nn.Linear with output = [gate|up] concatenated)
    - config with hidden_size, intermediate_size
    """

    def adapt(self):
        """Split gate_up_proj into separate gate_proj, up_proj Linear layers."""
        config = self.config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        has_bias = self.gate_up_proj.bias is not None

        gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=has_bias)
        up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=has_bias)

        gate_proj.weight.data.copy_(self.gate_up_proj.weight[:intermediate_size, :])
        up_proj.weight.data.copy_(self.gate_up_proj.weight[intermediate_size:, :])

        if has_bias:
            gate_proj.bias.data.copy_(self.gate_up_proj.bias[:intermediate_size])
            up_proj.bias.data.copy_(self.gate_up_proj.bias[intermediate_size:])

        del self.gate_up_proj
        self.gate_proj = gate_proj
        self.up_proj = up_proj


# ============================================================================
# Phi-3 SplitFusedLayers Implementation
# ============================================================================


class SplitFusedPhi3Attention(SplitQKVAttentionMixin, modeling_phi3.Phi3Attention):
    """Phi3Attention with split q/k/v projections instead of fused qkv_proj."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SplitFusedPhi3MLP(SplitGateUpMLPMixin, modeling_phi3.Phi3MLP):
    """Phi3MLP with split gate/up projections instead of fused gate_up_proj."""

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        gate = self.gate_proj(hidden_states)
        up_states = self.up_proj(hidden_states)

        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


@contextlib.contextmanager
def enable_split_fused_phi3():
    """Context manager to temporarily replace Phi3Attention and Phi3MLP with split versions."""
    original_attention = modeling_phi3.Phi3Attention
    original_mlp = modeling_phi3.Phi3MLP

    modeling_phi3.Phi3Attention = SplitFusedPhi3Attention
    modeling_phi3.Phi3MLP = SplitFusedPhi3MLP

    yield

    modeling_phi3.Phi3Attention = original_attention
    modeling_phi3.Phi3MLP = original_mlp


class Phi3SplitFusedLayersMixin:
    """Mixin to enable split fused layers for Phi3 models."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        with enable_split_fused_phi3():
            model = super().instantiate_model(*args, **kwargs)

        for module in model.modules():
            if isinstance(module, AdaptedModule):
                module.adapt()

        return model


# ============================================================================
# Registered SplitFusedLayers Adaptations
# ============================================================================


@YAMLConfigParser.register_adaptation(
    "SplitFusedLayers", model_type="phi3", exclusive=False
)
class Phi3SplitFusedLayersAdaptation(Phi3SplitFusedLayersMixin):
    """SplitFusedLayers adaptation for Phi3 models."""

    pass
