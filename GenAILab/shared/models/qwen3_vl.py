# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-3-VL model class"""

import typing
import torch

from transformers import AutoConfig, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.models.qwen3_vl import modeling_qwen3_vl

from GenAILab.shared.models.base import VLM
from GenAILab.shared.models.generator import VLM_Generator
from GenAILab.shared.models.utils.layer_cache import LayerCacheDescriptor
from GenAILab.shared.models.utils.model_utils import (
    PositionIdContext,
    compute_vision_input_shapes,
)


class Qwen_3_VL(VLM):
    """Generic quantized Qwen 3 VL"""

    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

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
        return modeling_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
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
        cls, model, context_length, sequence_length, layer_cache_descriptors
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

        dummy_visual_pos_masks = torch.zeros((1, sequence_length), dtype=torch.int)
        # Set some token in the middle to be an image token, which should force the model to consider visual input
        dummy_visual_pos_masks[0, sequence_length // 2] = 1
        dummy_visual_pos_masks = dummy_visual_pos_masks.to(dtype=torch.bool)

        dummy_deepstack_visual_embeds = [
            torch.zeros(1, model.config.hidden_size) for _ in range(3)
        ]

        return (
            *assembled_dummy_inputs,
            dummy_visual_pos_masks,
            dummy_deepstack_visual_embeds,
        )

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
            torch.zeros((1, 2048), dtype=torch.int),
        )

    def generate_position_ids(self, *args, **kwargs):
        ctx = PositionIdContext(self.config, modeling_qwen3_vl.Qwen3VLModel)
        position_ids, *_ = modeling_qwen3_vl.Qwen3VLModel.get_rope_index(
            ctx, *args, **kwargs
        )
        return position_ids.to(dtype=torch.int32)

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> tuple[str, ...]:
        return VLM.get_backbone_input_names(layer_cache_descriptors) + (
            "visual_pos_masks",
            "deepstack_visual_embeds",
        )

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return ("pixel_values", "image_grid_thw")

    @staticmethod
    def get_visual_output_names() -> tuple[str, ...]:
        return ("image_embeddings", "visual_pos_masks", "deepstack_visual_embeds")

    @staticmethod
    def get_generator_cls() -> type[VLM_Generator]:
        return Qwen3VL_Generator


class Qwen3VLVisualWrapper(torch.nn.Module):
    # Not moving this into shared code since this is pretty model specific
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_outputs = self.visual(
                pixel_values, grid_thw=image_grid_thw, return_dict=True
            )
            return (
                vision_outputs.pooler_output,
                mask[..., 0],
                vision_outputs.deepstack_features,
            )
        else:
            return None, mask[..., 0], []


class Qwen3VL_Generator(VLM_Generator):
    @staticmethod
    def slice_inputs_for_inference(
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_length: int,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> typing.Generator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None], None, None
    ]:
        deepstack = kwargs.pop("deepstack_visual_embeds", [])
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)

        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            position_ids_slice = (
                position_ids[..., max(0, idx - sequence_length) : idx]
                if position_ids is not None
                else None
            )

            extra_kwargs = {}
            if visual_pos_masks is not None:
                previously_consumed_visual_pos_masks_slice = visual_pos_masks[
                    :, : max(0, idx - sequence_length)
                ]
                num_previously_consumed_visual_tokens = (
                    previously_consumed_visual_pos_masks_slice.sum().item()
                )
                visual_pos_masks_slice = visual_pos_masks[
                    :, max(0, idx - sequence_length) : idx
                ]
                num_visual_tokens_to_consume = visual_pos_masks_slice.sum().item()

                start_idx = num_previously_consumed_visual_tokens
                stop_idx = (
                    num_previously_consumed_visual_tokens + num_visual_tokens_to_consume
                )
                sliced_deepstack = [
                    deepstack_i[
                        ...,
                        start_idx:stop_idx,
                        :,
                    ]
                    for deepstack_i in deepstack
                ]

                extra_kwargs |= {
                    "visual_pos_masks": visual_pos_masks_slice,
                    "deepstack_visual_embeds": sliced_deepstack,
                }

            yield (
                inputs[:, max(0, idx - sequence_length) : idx],
                attention_mask[:, max(0, idx - sequence_length) : idx],
                position_ids_slice,
                (kwargs or {}) | extra_kwargs,
            )

    @classmethod
    def prepare_inputs(
        cls,
        model: torch.nn.Module,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        pad_token: int = 0,
        attention_mask_min: int = -100,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        if visual_pos_masks is not None:
            visual_pos_mask_padding_size = sequence_length - visual_pos_masks.shape[1]
            if visual_pos_mask_padding_size > 0:
                visual_pos_masks_padding = torch.zeros(
                    (visual_pos_masks.shape[0], visual_pos_mask_padding_size),
                    dtype=visual_pos_masks.dtype,
                    device=visual_pos_masks.device,
                )
                visual_pos_masks = torch.cat(
                    (visual_pos_masks_padding, visual_pos_masks), dim=-1
                )

        return super().prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            sequence_length=sequence_length,
            context_length=context_length,
            pad_token=pad_token,
            attention_mask_min=attention_mask_min,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            **kwargs,
        )
