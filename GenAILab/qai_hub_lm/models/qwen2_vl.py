# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-2.5-VL model class"""

import torch

from transformers import AutoConfig, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

from .base import VLM
from .generator import VLM_Generator
from .utils.compat import PositionIdContext
from .utils.layer_cache import LayerCacheDescriptor


def compute_vision_input_shapes(
    image_size: tuple[int, int],
    vision_config,
) -> tuple[int, int, int, int]:
    """Compute vision encoder input shapes from a target image size.

    Args:
        image_size: Target (width, height) that images will be resized to.
            Follows PIL convention.
        vision_config: HF vision config with ``patch_size``,
            ``spatial_merge_size``, ``temporal_patch_size``, and
            ``in_channels`` attributes.

    Returns:
        (num_patches, pixel_dim, h_patches, w_patches)
    """
    w, h = image_size
    patch_size = vision_config.patch_size
    merge_size = vision_config.spatial_merge_size
    temporal_patch_size = vision_config.temporal_patch_size
    in_channels = vision_config.in_channels

    factor = patch_size * merge_size
    h_patches = (h // factor) * merge_size
    w_patches = (w // factor) * merge_size

    num_patches = h_patches * w_patches
    pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size

    return num_patches, pixel_dim, h_patches, w_patches


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

        prepared = VLM_Generator.prepare_inputs(
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
        return tuple(prepared.values())

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
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        num_new_tokens = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int32)

        has_multimodal = (
            kwargs.get("image_grid_thw") is not None
            or kwargs.get("video_grid_thw") is not None
        )

        if has_multimodal:
            trimmed_mask = attention_mask[:, -num_new_tokens:]
            ctx = PositionIdContext(self.config, modeling_qwen2_5_vl.Qwen2_5_VLModel)
            position_ids, rope_deltas = (
                modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index(
                    ctx,
                    *args,
                    input_ids=input_ids.long(),
                    attention_mask=trimmed_mask,
                    **kwargs,
                )
            )
            self._rope_deltas = rope_deltas
            return position_ids[..., -num_new_tokens:].to(dtype=torch.int32)
        elif (
            num_new_tokens == 1
            and hasattr(self, "_rope_deltas")
            and self._rope_deltas is not None
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids[:, -num_new_tokens:]
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            position_ids = position_ids + self._rope_deltas.unsqueeze(-1)
            return position_ids.to(dtype=torch.int32)
        else:
            trimmed_mask = attention_mask[:, -num_new_tokens:]
            ctx = PositionIdContext(self.config, modeling_qwen2_5_vl.Qwen2_5_VLModel)
            position_ids, rope_deltas = (
                modeling_qwen2_5_vl.Qwen2_5_VLModel.get_rope_index(
                    ctx,
                    *args,
                    input_ids=input_ids.long(),
                    attention_mask=trimmed_mask,
                    **kwargs,
                )
            )
            self._rope_deltas = rope_deltas
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


def enable_fast_exportable_vision_attention():
    """Re-export from transforms for backwards compatibility."""
    from GenAILab.qai_hub_lm.transforms.fast_exportable import (
        enable_qwen2_vl_fast_exportable_vision_attention,
    )

    return enable_qwen2_vl_fast_exportable_vision_attention()
