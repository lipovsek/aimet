# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-2.5-VL model class"""

import contextlib
import torch

from transformers import AutoConfig, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from GenAILab.qai_hub_lm.models.base import VLM
from GenAILab.qai_hub_lm.models.generator import VLM_Generator
from GenAILab.qai_hub_lm.utils.model_utils import (
    PositionIdContext,
    compute_vision_input_shapes,
)

from GenAILab.qai_hub_lm.utils.layer_cache import LayerCacheDescriptor


class Qwen_25_VL(VLM):
    """Generic quantized Qwen 2.5 VL"""

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    @classmethod
    def instantiate_model(cls, model_id: str, small_model=False) -> PreTrainedModel:
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
        return modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> ProcessorMixin:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID
        return AutoProcessor.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def get_sample_backbone_inputs(
        cls,
        model,
        context_length: int,
        sequence_length: int,
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        *args,
        **kwargs,
    ):
        dummy_inputs_embeds = torch.zeros(
            (1, sequence_length, model.config.hidden_size), dtype=torch.int
        )
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)
        dummy_position_ids = torch.zeros((3, 1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = VLM_Generator.prepare_inputs(
            model=model,
            input_ids=None,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            inputs_embeds=dummy_inputs_embeds,
            position_ids=dummy_position_ids,
            layer_cache_descriptors=layer_cache_descriptors,
        )
        return assembled_dummy_inputs

    @classmethod
    def get_sample_vision_inputs(cls, config, image_size=(512, 512)):
        num_patches, pixel_dim, h_patches, w_patches = compute_vision_input_shapes(
            image_size, config.vision_config
        )
        dummy_pixel_values = torch.ones((num_patches, pixel_dim), dtype=torch.float32)
        dummy_grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.int64)
        return (
            dummy_pixel_values,
            dummy_grid_thw,
            torch.Tensor(
                [
                    0,
                ]
            ),
        )

    def generate_position_ids(
        self,
        *args,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        num_new_tokens = input_ids.shape[1]
        attention_mask = attention_mask[:, -num_new_tokens:]

        ctx = PositionIdContext(self.config, modeling_qwen2_5_vl.Qwen2_5_VLModel)
        position_ids, *_ = modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index(
            ctx, *args, input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return position_ids[..., -num_new_tokens:].to(dtype=torch.int32)

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return ("pixel_values", "image_grid_thw")

    @staticmethod
    def get_visual_output_names() -> tuple[str, ...]:
        return ("image_embeddings",)


class Qwen2VLVisualWrapper(torch.nn.Module):
    # Not moving this into shared code since this is pretty model specific
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_outputs = self.visual(
                pixel_values, grid_thw=image_grid_thw, return_dict=True
            )
            return vision_outputs.pooler_output
        else:
            return None


#################################  Exportable Vision Attention for ONNX  #################################


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
    # Create position indices [total_length]
    positions = torch.arange(total_length, device=device)

    # Compute segment ID for each position using tensor ops
    # Position p is in segment i if cu_seqlens[i] <= p < cu_seqlens[i+1]
    boundaries = cu_seqlens[:-1].unsqueeze(1)  # [num_segments, 1]
    positions_expanded = positions.unsqueeze(0)  # [1, total_length]
    segment_ids = (positions_expanded >= boundaries).sum(dim=0) - 1  # [total_length]

    # Create mask: can attend only within same segment
    same_segment = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0)  # [seq, seq]

    # Convert to additive mask (0 = attend, -inf = block)
    mask = torch.where(same_segment, 0.0, float("-inf"))

    return mask.unsqueeze(0).unsqueeze(0).to(dtype)  # [1, 1, seq, seq]


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

        # Reshape for attention: [1, num_heads, seq_length, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Create block-diagonal attention mask from cu_seqlens
        # This preserves the windowed attention behavior without loops
        attention_mask = _create_block_diagonal_mask(
            cu_seqlens, seq_length, query_states.dtype, query_states.device
        )

        # Use the attention implementation specified in config
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
def enable_fast_exportable_vision_attention():
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
