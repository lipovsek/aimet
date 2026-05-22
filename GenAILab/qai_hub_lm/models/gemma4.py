# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Gemma4 shared VLM base class"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedModel, ProcessorMixin
from huggingface_hub import hf_hub_download

try:
    from transformers.models.gemma4 import modeling_gemma4
except ImportError:
    modeling_gemma4 = None

from GenAILab.qai_hub_lm.models.base import VLM
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from GenAILab.qai_hub_lm.models.utils.layer_cache import LayerCacheDescriptor


class SoftcappedLMHead(torch.nn.Module):
    """LM head wrapper that applies Gemma4's final_logit_softcapping after the linear.

    When softcap is None, acts as a transparent pass-through.
    """

    def __init__(self, lm_head: torch.nn.Module, softcap: float | None):
        super().__init__()
        self.lm_head = lm_head
        self.softcap = softcap

    @property
    def linear(self) -> torch.nn.Module:
        return self.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if self.softcap is not None:
            logits = logits / self.softcap
            logits = torch.tanh(logits)
            logits = logits * self.softcap
        return logits


class Gemma4VisionWrapper(torch.nn.Module):
    """Wraps Gemma4's vision_tower + embed_vision projector into a single traceable module.

    Inputs:  pixel_values       [B, num_patches, 3*patch_size^2]
             image_position_ids [B, num_patches, 2]
    Output:  image_embeddings   [total_image_tokens, text_hidden_size]
    """

    def __init__(self, vision_tower, embed_vision):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_out = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        return self.embed_vision(inputs_embeds=vision_out.last_hidden_state)


class Gemma4_VLM_Generator(VLM_Generator):
    """VLM_Generator subclass for Gemma4.

    Gemma4 uses ``pixel_values`` + ``image_position_ids`` (one entry per image),
    rather than the Qwen-style ``pixel_values`` + ``image_grid_thw``.
    """

    def __init__(self, *args, embed_tokens_per_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed_tokens_per_layer = embed_tokens_per_layer

    @staticmethod
    def slice_inputs_for_inference(
        inputs, attention_mask, sequence_length, position_ids=None, **kwargs
    ):
        per_layer_inputs = kwargs.pop("per_layer_inputs", None)
        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            start = max(0, idx - sequence_length)
            input_slice = inputs[:, start:idx]
            mask_slice = attention_mask[:, start:idx]
            pos_slice = (
                position_ids[..., start:idx] if position_ids is not None else None
            )
            kw_slice = dict(kwargs)
            slice_len = input_slice.shape[1]
            pad_len = sequence_length - slice_len
            if per_layer_inputs is not None:
                ple_slice = per_layer_inputs[:, start:idx]
                if pad_len > 0:
                    pad_shape = (ple_slice.shape[0], pad_len) + ple_slice.shape[2:]
                    ple_slice = torch.cat(
                        [
                            torch.zeros(
                                pad_shape,
                                dtype=ple_slice.dtype,
                                device=ple_slice.device,
                            ),
                            ple_slice,
                        ],
                        dim=1,
                    )
                kw_slice["per_layer_inputs"] = ple_slice
            yield input_slice, mask_slice, pos_slice, kw_slice

    def _compute_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute PLE token-identity embeddings from input_ids.

        Uses the stored ``embed_tokens_per_layer`` embedding table directly.
        Applies sqrt(ple_dim) scaling unconditionally since the embedding may be
        a plain nn.Embedding (e.g. after cache deserialization) that lacks the
        built-in Gemma4TextScaledWordEmbedding scale factor.
        """
        text_config = self.config.text_config
        ple_dim = text_config.hidden_size_per_layer_input
        num_layers = text_config.num_hidden_layers

        pad_token_id = text_config.pad_token_id
        ple_input_ids = input_ids.clone()
        ple_input_ids[input_ids == self.config.image_token_id] = pad_token_id

        with torch.no_grad():
            per_layer_inputs = self._embed_tokens_per_layer(ple_input_ids)
        per_layer_inputs = per_layer_inputs.reshape(
            *ple_input_ids.shape, num_layers, ple_dim
        )
        if not isinstance(
            self._embed_tokens_per_layer, modeling_gemma4.Gemma4TextScaledWordEmbedding
        ):
            per_layer_inputs = per_layer_inputs * (ple_dim**0.5)
        return per_layer_inputs

    def fuse_text_image_video(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        text_config = self.config.text_config
        hidden_size = text_config.hidden_size

        inputs_embeds = self.embedding(input_ids)
        if not isinstance(
            self.embedding, modeling_gemma4.Gemma4TextScaledWordEmbedding
        ):
            inputs_embeds = inputs_embeds * (hidden_size**0.5)
        per_layer_inputs = self._compute_per_layer_inputs(input_ids)

        image_mask_3d = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

        if pixel_values is not None:
            all_embeddings = []
            num_images = pixel_values.shape[0]
            for i in range(num_images):
                pv_i = pixel_values[i].unsqueeze(0)
                pid_i = image_position_ids[i].unsqueeze(0)
                emb_i = self.vision_model(pv_i, pid_i)
                all_embeddings.append(emb_i)
            image_embeddings = torch.cat(all_embeddings, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_3d, image_embeddings
            )

        mm_token_type_ids = torch.zeros_like(input_ids)
        if pixel_values is not None:
            mm_token_type_ids[input_ids == self.config.image_token_id] = 1

        return (
            inputs_embeds,
            mm_token_type_ids,
            {"per_layer_inputs": per_layer_inputs},
        )

    def _prefill_visual(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        if pixel_values is None:
            return
        num_images = pixel_values.shape[0]
        for i in range(num_images):
            yield (pixel_values[i].unsqueeze(0), image_position_ids[i].unsqueeze(0))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        kwargs.pop("mm_token_type_ids", None)
        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        return Generator.forward(
            self,
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=None,
            **{**kwargs, **extra_kwargs},
        )

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        if self._visual_quantization_mode:
            yield from self._prefill_visual(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
                **kwargs,
            )
            return

        inputs_embeds, mm_token_type_ids, extra_kwargs = self.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        yield from Generator.prefill(
            self,
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=None,
            **{**kwargs, **extra_kwargs},
        )


class Gemma4_VLM(VLM):
    """Shared Gemma4 VLM base (framework-agnostic)."""

    DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"

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
            llm_config.text_config.num_hidden_layers = 2
            if (
                hasattr(llm_config.text_config, "layer_types")
                and llm_config.text_config.layer_types is not None
            ):
                llm_config.text_config.layer_types = llm_config.text_config.layer_types[
                    :2
                ]
        return modeling_gemma4.Gemma4ForConditionalGeneration.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> ProcessorMixin:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        chat_template_path = hf_hub_download(model_id, "chat_template.jinja")
        with open(chat_template_path) as f:
            return AutoProcessor.from_pretrained(
                model_id, use_fast=True, trust_remote_code=True, chat_template=f.read()
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
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        ple_dim = model.config.hidden_size_per_layer_input

        dummy_inputs_embeds = torch.zeros(
            (1, sequence_length, hidden_size), dtype=torch.float32
        )
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)
        dummy_per_layer_inputs = torch.zeros(
            (1, sequence_length, num_layers, ple_dim), dtype=torch.float32
        )

        prepared = Gemma4_VLM.get_generator_cls().prepare_inputs(
            model=model,
            input_ids=None,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            inputs_embeds=dummy_inputs_embeds,
            layer_cache_descriptors=layer_cache_descriptors,
            per_layer_inputs=dummy_per_layer_inputs,
        )
        return tuple(prepared.values())

    @classmethod
    def get_sample_vision_inputs(cls, config, image_size=None):
        """Dummy inputs for Gemma4 vision QuantSim.

        Gemma4's image processor always pads to 2520 patches
        (image_seq_length=280 * pooling_kernel_size^2=9).
        """
        vcfg = config.vision_config
        patch_dim = 3 * vcfg.patch_size**2
        num_patches = 2520
        dummy_pixel_values = torch.zeros(
            (1, num_patches, patch_dim), dtype=torch.float32
        )
        dummy_position_ids = torch.zeros((1, num_patches, 2), dtype=torch.int64)
        return (dummy_pixel_values, dummy_position_ids)

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> tuple[str, ...]:
        from GenAILab.qai_hub_lm.models.utils.layer_cache import (
            attention_mask_input_names,
            cache_state_names,
        )

        return tuple(
            ["inputs_embeds"]
            + attention_mask_input_names(layer_cache_descriptors)
            + ["position_ids"]
            + cache_state_names(layer_cache_descriptors, "in")
            + ["per_layer_inputs"]
        )

    @staticmethod
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        from GenAILab.qai_hub_lm.models.utils.layer_cache import (
            AttentionType,
            attention_mask_input_names,
        )

        axes: dict[str, dict[int, str]] = {
            "inputs_embeds": {1: "sequence_length"},
            "position_ids": {1: "sequence_length"},
            "logits": {1: "sequence_length"},
            "per_layer_inputs": {1: "sequence_length"},
        }
        for name in attention_mask_input_names(layer_cache_descriptors):
            axes[name] = {2: "sequence_length"}
        for desc in layer_cache_descriptors:
            i = desc.layer_idx
            if desc.attention_type == AttentionType.LINEAR:
                continue
            axes[f"past_key_{i}_in"] = {2: "kv_cache_length"}
            axes[f"past_value_{i}_in"] = {2: "kv_cache_length"}
        return axes

    @classmethod
    def instantiate_position_processor(cls):
        return None

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return ("pixel_values", "image_position_ids")

    @staticmethod
    def get_visual_output_names() -> tuple[str, ...]:
        return ("image_embeddings",)

    @staticmethod
    def get_generator_cls():
        return Gemma4_VLM_Generator
