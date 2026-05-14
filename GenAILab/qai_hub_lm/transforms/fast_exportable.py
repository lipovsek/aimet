# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""FastExportable adaptations for VLM models.

Replaces loop-based vision attention with mask-based equivalents that produce
clean ONNX graphs.
"""

import contextlib
import types

import torch

from transformers import PreTrainedModel
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from GenAILab.bench.yaml_config_parser import YAMLConfigParser


# ============================================================================
# Shared utilities
# ============================================================================


def _create_block_diagonal_mask(
    cu_seqlens: torch.Tensor,
    total_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a block-diagonal attention mask from cumulative sequence lengths.

    Each segment defined by cu_seqlens can only attend within itself.
    Uses pure tensor operations for ONNX compatibility.

    Args:
        cu_seqlens: Cumulative lengths [0, len1, len1+len2, ..., total]
        total_length: Total sequence length
        dtype: Output dtype
        device: Output device

    Returns:
        Additive attention mask [1, 1, total_length, total_length]
        0 for allowed positions, -inf for blocked positions
    """
    positions = torch.arange(total_length, device=device)

    boundaries = cu_seqlens[:-1].unsqueeze(1)  # [num_segments, 1]
    positions_expanded = positions.unsqueeze(0)  # [1, total_length]
    segment_ids = (positions_expanded >= boundaries).sum(dim=0) - 1  # [total_length]

    same_segment = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0)  # [seq, seq]

    mask = torch.where(same_segment, 0.0, float("-inf"))

    return mask.unsqueeze(0).unsqueeze(0).to(dtype)  # [1, 1, seq, seq]


# ============================================================================
# Qwen 2.5 VL
# ============================================================================


class FastExportableQwen2_5_VLVisionAttention(
    modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention
):
    """
    Vision attention that uses attention masks instead of loop-based splitting.

    This produces a clean ONNX graph while preserving the windowed attention
    behavior defined by cu_seqlens.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen2_5_vl.apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_mask = _create_block_diagonal_mask(
            cu_seqlens, seq_length, query_states.dtype, query_states.device
        )

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        else:
            attention_interface = modeling_qwen2_5_vl.eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


@contextlib.contextmanager
def enable_qwen2_vl_fast_exportable_vision_attention():
    """
    Context manager that temporarily replaces Qwen2_5_VLVisionAttention with
    the exportable version that uses attention masks instead of loop-based splitting.
    """
    original = modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention
    modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention = (
        FastExportableQwen2_5_VLVisionAttention
    )

    try:
        yield
    finally:
        modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention = original


# ============================================================================
# Qwen 3 VL
# ============================================================================


class FastExportableQwen3VLVisionAttention(modeling_qwen3_vl.Qwen3VLVisionAttention):
    """
    Vision attention that uses attention masks instead of loop-based splitting.

    This produces a clean ONNX graph while preserving the windowed attention
    behavior defined by cu_seqlens.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_mask = _create_block_diagonal_mask(
            cu_seqlens, seq_length, query_states.dtype, query_states.device
        )

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        else:
            attention_interface = modeling_qwen3_vl.eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


def _exportable_deepstack_process(
    self,
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    visual_embeds: torch.Tensor,
) -> torch.Tensor:
    """Export-friendly replacement for ``_deepstack_process``.

    HF's implementation uses boolean indexing (``hidden_states[mask, :]``)
    which creates a data-dependent intermediate shape that
    ``torch.export.draft_export`` cannot trace.  This version uses
    ``cumsum`` + ``gather`` so all shapes remain static.
    """
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

    mask_int = visual_pos_masks.int()

    cumsum = mask_int.cumsum(dim=-1)  # [batch, seq_len]

    gather_idx = cumsum * mask_int  # [batch, seq_len]
    gather_idx = gather_idx.unsqueeze(-1).expand(
        -1, -1, hidden_states.shape[-1]
    )  # [batch, seq_len, hidden]

    zero_row = torch.zeros(
        1,
        hidden_states.shape[-1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    visual_embeds_ext = torch.cat([zero_row, visual_embeds], dim=0)
    visual_embeds_ext = visual_embeds_ext.unsqueeze(0).expand(
        hidden_states.shape[0], -1, -1
    )  # [batch, num_visual + 1, hidden]

    visual_full = torch.gather(visual_embeds_ext, 1, gather_idx)

    return hidden_states + visual_full


@contextlib.contextmanager
def enable_qwen3_vl_fast_exportable_vision_attention():
    """
    Context manager that temporarily replaces Qwen3VLVisionAttention with
    the exportable version that uses attention masks instead of loop-based splitting.
    """
    original_attn = modeling_qwen3_vl.Qwen3VLVisionAttention
    modeling_qwen3_vl.Qwen3VLVisionAttention = FastExportableQwen3VLVisionAttention

    try:
        yield
    finally:
        modeling_qwen3_vl.Qwen3VLVisionAttention = original_attn


# ============================================================================
# Registered Adaptations
# ============================================================================


@YAMLConfigParser.register_adaptation(
    "FastExportable", model_type="qwen2_5_vl", required_for_export=True
)
class Qwen2VLFastExportableAdaptation:
    """FastExportable adaptation for Qwen2 VL models."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        with enable_qwen2_vl_fast_exportable_vision_attention():
            model = super().instantiate_model(*args, **kwargs)
        return model


@YAMLConfigParser.register_adaptation(
    "FastExportable", model_type="qwen3_vl", required_for_export=True
)
class Qwen3VLFastExportableAdaptation:
    """FastExportable adaptation for Qwen3 VL models."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        with enable_qwen3_vl_fast_exportable_vision_attention():
            model = super().instantiate_model(*args, **kwargs)

        text_model = model.model.language_model
        text_model._deepstack_process = types.MethodType(
            _exportable_deepstack_process, text_model
        )
        return model
