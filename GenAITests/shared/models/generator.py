# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM Generator class to restore HF API on models with static shape constraints"""

from __future__ import annotations

import itertools
import math
from typing import Union
import types

import torch
import transformers
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """
    Combine past_key_vals with new_key_vals and clip them so there are no more than `length` tokens worth of context.
    """
    ret = []

    # If there are no past_key_vals create some empty ones in the correct shape
    if len(past_key_vals) == 0:
        for i in range(0, len(new_key_vals), 2):
            key_shape = new_key_vals[i].shape
            key_shape = (key_shape[0], key_shape[1], 0, key_shape[3])
            past_key_vals.append(torch.zeros(key_shape, device=device))

            value_shape = new_key_vals[i + 1].shape
            value_shape = (value_shape[0], value_shape[1], 0, value_shape[3])
            past_key_vals.append(torch.zeros(value_shape, device=device))

    # If there are no new_key_vals create some empty ones in the correct shape
    if len(new_key_vals) == 0:
        for i in range(0, len(past_key_vals), 2):
            key_shape = past_key_vals[i].shape
            key_shape = (key_shape[0], key_shape[1], 0, key_shape[3])
            new_key_vals.append(torch.zeros(key_shape, device=device))

            value_shape = past_key_vals[i + 1].shape
            value_shape = (value_shape[0], value_shape[1], 0, value_shape[3])
            new_key_vals.append(torch.zeros(value_shape, device=device))

    # Key and Values are concatenated on batch dimension
    for i in range(0, len(past_key_vals), 2):
        key_cache = torch.cat(
            [past_key_vals[i].to(device), new_key_vals[i].to(device)],
            dim=2,
        )
        key_cache = key_cache[:, :, -length:, :]
        val_cache = torch.cat(
            [
                past_key_vals[i + 1].to(device),
                new_key_vals[i + 1].to(device),
            ],
            dim=2,
        )
        val_cache = val_cache[:, :, -length:, :]

        ret.append(key_cache.to(dtype=dtype))
        ret.append(val_cache.to(dtype=dtype))
    return ret


class Generator(GenerationMixin, torch.nn.Module):
    """
    Helper class to restore HuggingFace LLM API to Torch and ONNX models with static shape requirements, including
    the `forward` and `generate` APIs
    """

    def __init__(
        self,
        model,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int,
        context_length: int,
        config: Union[PretrainedConfig | None] = None,
        attention_mask_min: int = -100,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.generation_config = None
        self._config = config
        self.attention_mask_min = attention_mask_min

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        if self._config is not None:
            return self._config
        return self.model.config

    @property
    def main_input_name(self) -> str:
        return "input_ids"

    @property
    def _supports_cache_class(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return self.model.device

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | DynamicCache | None]:
        """
        Overridden prepare_inputs_for_generation function to enable Huggingface generate() on models with static
        graph constraints
        """

        # We need a way to ensure that all the previous tokens that have already been consumed are stripped out of the
        # input ids

        # If past_key_values is None, this indicates that this `prepare_inputs_for_generation()` is being called for
        # the first time, and nothing should be stripped out of `input_ids`. In other cases though, the number of tokens
        # already inside `past_key_values` indicates how many tokens should be stripped out of `input_ids`

        # Notes: `input_ids`, `attention_mask`, `past_key_values` should NOT have static shape requirements imposed on
        # them by the time they reach this function. That is, in order for this to work, the static shape padding and
        # truncation must happen directly in the model `forward` function

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        num_processed_tokens = (
            past_key_values.get_seq_length()
            if not isinstance(past_key_values, tuple)
            else past_key_values[0][1].shape[-2]
        )

        inputs = (
            {"input_ids": input_ids[:, num_processed_tokens:]}
            if input_ids is not None
            else {"inputs_embeds": inputs_embeds[:, num_processed_tokens:, :]}
        )
        return inputs | {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def slice_inputs_for_inference(
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_length: int,
        position_ids: torch.Tensor | None = None,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None], None, None]:
        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            position_ids_slice = (
                position_ids[..., max(0, idx - sequence_length) : idx]
                if position_ids is not None
                else None
            )
            yield (
                inputs[:, max(0, idx - sequence_length) : idx],
                attention_mask[:, max(0, idx - sequence_length) : idx],
                position_ids_slice,
            )

    @staticmethod
    def get_input_names(num_layers: int) -> tuple[str, ...]:
        names = ["input_ids", "attention_mask", "position_ids"]
        kv_names = zip(
            [f"past_key_{i}_in" for i in range(num_layers)],
            [f"past_value_{i}_in" for i in range(num_layers)],
        )

        return tuple(names + list(itertools.chain.from_iterable(kv_names)))

    @staticmethod
    def get_output_names(num_layers: int) -> tuple[str, ...]:
        names = ["logits"]
        kv_names = zip(
            [f"past_key_{i}_out" for i in range(num_layers)],
            [f"past_value_{i}_out" for i in range(num_layers)],
        )

        return tuple(names + list(itertools.chain.from_iterable(kv_names)))

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
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Prepare provided inputs for model forward pass with static graph constraints"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        input_tokens = input_ids if input_ids is not None else inputs_embeds
        input_tokens = input_tokens.to(
            dtype=torch.int32 if input_ids is not None else torch.float32
        )

        device = input_tokens.device
        batch_size = input_tokens.shape[0]
        input_length = input_tokens.shape[1]

        # Create attention mask if one is not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=device,
            )

        # Pad input tokens and attention mask to [batch_size, sequence_length] (necessary for static shape requirements)
        if input_ids is not None:
            input_tokens_extension = torch.full(
                (batch_size, sequence_length - input_length),
                fill_value=pad_token,
                dtype=input_tokens.dtype,
                device=device,
            )
        else:
            embedding_dim = input_tokens.shape[2]
            input_tokens_extension = torch.zeros(
                (batch_size, sequence_length - input_length, embedding_dim),
                dtype=input_tokens.dtype,
                device=device,
            )

        padded_input_tokens = torch.cat((input_tokens_extension, input_tokens), dim=1)
        padded_attention_mask = torch.cat(
            (
                torch.zeros(
                    (batch_size, sequence_length - input_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

        # Create dummy KV cache
        head_dim = (
            model.config.head_dim
            if hasattr(model.config, "head_dim") and model.config.head_dim is not None
            else model.config.hidden_size // model.config.num_attention_heads
        )
        kv_shape = (
            batch_size,
            model.config.num_key_value_heads,
            context_length - sequence_length,
            head_dim,
        )
        dummy_past_key_values = (
            [
                torch.zeros(kv_shape, device=device),
            ]
            * model.config.num_hidden_layers
            * 2
        )

        current_key_value_length = (
            past_key_values[0].shape[-2]
            if past_key_values and len(past_key_values) > 0
            else 0
        )
        key_value_padding_length = (
            context_length - sequence_length
        ) - current_key_value_length

        # Join input past_key_values with dummy_past_key_values, and clip all padding values that go over the max context
        padded_past_key_values = get_past_keyval_with_shift(
            past_key_vals=dummy_past_key_values,
            new_key_vals=past_key_values,
            length=context_length - sequence_length,
            device=device,
            dtype=model.dtype,
        )

        # Mask out dummy entries in KV cache
        kv_cache_attention_mask = torch.cat(
            (
                torch.zeros((batch_size, key_value_padding_length)),
                torch.ones((batch_size, current_key_value_length)),
            ),
            dim=-1,
        ).to(device=device)
        padded_attention_mask = torch.cat(
            (kv_cache_attention_mask, padded_attention_mask), dim=-1
        )

        # Convert attention mask from 1D to 4D and clip values
        attention_mask_converter = AttentionMaskConverter(True)
        cm_attention_mask = attention_mask_converter.to_4d(
            padded_attention_mask,
            query_length=sequence_length,
            key_value_length=context_length,
            dtype=torch.float32,
        )
        cm_attention_mask = cm_attention_mask.clip(attention_mask_min, 0)

        # Compute or pad position_ids
        if position_ids is None:
            # Compute position_ids from attention mask
            position_ids = (
                torch.cumsum(padded_attention_mask, dim=1, dtype=torch.int32) - 1
            )
            position_ids = position_ids.clip(0, context_length - 1)
            position_ids = position_ids[:, -sequence_length:]
        else:
            # Pad provided position_ids to sequence_length (pad in last dimension)
            padding_length = sequence_length - position_ids.shape[-1]
            if padding_length > 0:
                # Build padding shape dynamically - works for both 2D and 3D tensors
                pad_shape = list(position_ids.shape)
                pad_shape[-1] = padding_length
                position_ids_padding = torch.zeros(
                    pad_shape,
                    dtype=position_ids.dtype,
                    device=device,
                )
                position_ids = torch.cat((position_ids, position_ids_padding), dim=-1)

        return (
            padded_input_tokens,
            cm_attention_mask.to(dtype=model.dtype),
            position_ids,
            *padded_past_key_values,
        )

    def combine_local_and_global_outputs(
        self,
        num_valid_input_tokens: int,
        local_outputs: tuple[torch.Tensor, ...],
        global_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]],
    ):
        # strip logits corresponding to padding tokens
        local_logits = local_outputs[0]
        local_logits = torch.narrow(
            local_logits,
            1,
            local_logits.shape[1] - num_valid_input_tokens,
            num_valid_input_tokens,
        )

        # concatenate logits from local inference to global output
        global_outputs["logits"] = (
            torch.cat((global_outputs["logits"], local_logits), dim=1)
            if "logits" in global_outputs
            else local_logits
        )

        # strip KV cache corresponding to padding tokens
        local_past_key_values = get_past_keyval_with_shift(
            past_key_vals=[],
            new_key_vals=list(local_outputs[1:]),
            length=num_valid_input_tokens,
            device=self.device,
        )

        # shift global KV cache, concatenate local KV cache
        current_key_value_length = (
            global_outputs["past_key_values"][0].shape[-2]
            if global_outputs["past_key_values"]
            else 0
        )
        global_outputs["past_key_values"] = get_past_keyval_with_shift(
            past_key_vals=global_outputs["past_key_values"],
            new_key_vals=local_past_key_values,
            length=min(
                current_key_value_length + num_valid_input_tokens,
                self.context_length - self.sequence_length,
            ),
            device=self.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        global_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]] = {
            "past_key_values": []
            if past_key_values is None or past_key_values.get_seq_length() == 0
            else list(itertools.chain.from_iterable(past_key_values.to_legacy_cache()))
        }

        for (
            input_slice,
            attention_mask_slice,
            position_ids_slice,
        ) in self.slice_inputs_for_inference(
            input_tokens, attention_mask, self.sequence_length, position_ids
        ):
            prepared_inputs = self.prepare_inputs(
                model=self.model,
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=global_outputs["past_key_values"],
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                pad_token=getattr(self.tokenizer, "eos_token_id", 0),
                attention_mask_min=self.attention_mask_min,
                inputs_embeds=input_slice if inputs_embeds is not None else None,
                position_ids=position_ids_slice,
                **kwargs,
            )

            local_outputs = self.model(*prepared_inputs)

            self.combine_local_and_global_outputs(
                input_slice.shape[1], local_outputs, global_outputs
            )

        # make sure all outputs are on the correct device
        # the underlying mock_torch_onnx_inference function does not necessarily move outputs back to CUDA
        assert isinstance(global_outputs["logits"], torch.Tensor)
        logits = global_outputs["logits"].to(device=self.device)
        past_key_values_list = list(
            map(
                lambda tensor: tensor.to(device=self.device),
                global_outputs["past_key_values"],
            )
        )

        # Convert KV Cache outputs into HF DynamicCache
        past_key_values = DynamicCache()
        past_key_values.key_cache = past_key_values_list[::2]
        past_key_values.value_cache = past_key_values_list[1::2]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        input_tokens = input_ids if input_ids is not None else inputs_embeds

        # Create attention mask if one does not exist
        if attention_mask is None:
            batch_size = input_tokens.shape[0]
            input_length = input_tokens.shape[1]
            attention_mask = torch.ones(
                (batch_size, input_length),
                dtype=torch.int32,
                device=input_tokens.device,
            )

        # slice input tokens and attention mask to drop last few tokens
        total_num_inferences = math.ceil(input_tokens.shape[1] / self.sequence_length)
        num_tokens_to_preconsume = (total_num_inferences - 1) * self.sequence_length

        input_tokens_to_preconsume = input_tokens[:, :num_tokens_to_preconsume]
        attention_mask_to_preconsume = attention_mask[:, :num_tokens_to_preconsume]
        # Slice position_ids to match the tokens to preconsume (slice last dimension)
        position_ids_to_preconsume = (
            position_ids[..., :num_tokens_to_preconsume]
            if position_ids is not None
            else None
        )

        preconsumed_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]] = {
            "past_key_values": []
            if past_key_values is None or past_key_values.get_seq_length() == 0
            else list(itertools.chain.from_iterable(past_key_values.to_legacy_cache()))
        }

        for (
            input_slice,
            attention_mask_slice,
            position_ids_slice,
        ) in self.slice_inputs_for_inference(
            input_tokens_to_preconsume,
            attention_mask_to_preconsume,
            self.sequence_length,
            position_ids_to_preconsume,
        ):
            prepared_inputs = self.prepare_inputs(
                model=self.model,
                input_ids=input_slice if input_ids is not None else None,
                attention_mask=attention_mask_slice,
                past_key_values=preconsumed_outputs["past_key_values"],
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                pad_token=getattr(self.tokenizer, "eos_token_id", 0),
                attention_mask_min=self.attention_mask_min,
                inputs_embeds=input_slice if inputs_embeds is not None else None,
                position_ids=position_ids_slice,
                **kwargs,
            )

            yield prepared_inputs

            local_outputs = self.model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                input_slice.shape[1],
                local_outputs,
                preconsumed_outputs,
            )

        remaining_input_tokens = input_tokens[:, num_tokens_to_preconsume:]
        remaining_attention_mask = attention_mask[:, num_tokens_to_preconsume:]
        # Slice remaining position_ids (slice last dimension)
        remaining_position_ids = (
            position_ids[..., num_tokens_to_preconsume:]
            if position_ids is not None
            else None
        )

        prefilled_inputs = self.prepare_inputs(
            model=self.model,
            input_ids=remaining_input_tokens if input_ids is not None else None,
            attention_mask=remaining_attention_mask,
            past_key_values=preconsumed_outputs["past_key_values"],
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            pad_token=getattr(self.tokenizer, "eos_token_id", 0),
            attention_mask_min=self.attention_mask_min,
            inputs_embeds=remaining_input_tokens if inputs_embeds is not None else None,
            position_ids=remaining_position_ids,
        )

        yield prefilled_inputs


class VLM_Generator(Generator):
    def __init__(
        self,
        backbone_model,
        vision_model,
        embedding,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int,
        context_length: int,
        position_id_processor=None,
        config: Union[PretrainedConfig | None] = None,
        attention_mask_min: int = -100,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=backbone_model,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            config=config,
            attention_mask_min=attention_mask_min,
            *args,
            **kwargs,
        )

        self.vision_model = vision_model
        self.embedding = embedding
        self.position_id_processor = (
            types.MethodType(position_id_processor, self)
            if position_id_processor
            else None
        )

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return "pixel_values", "image_grid_thw"

    @staticmethod
    def get_visual_output_names() -> tuple[str, ...]:
        return ("image_embeddings",)

    def fuse_text_image_video(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ):
        # 1) Convert input_ids to input embeddings using self.embedding
        inputs_embeds = self.embedding(input_ids)

        # 2) Process images and videos through vision model to get image embeddings
        if pixel_values is not None:
            image_embeddings = self.vision_model(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            ).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeddings)

        if pixel_values_videos is not None:
            video_embeddings = self.vision_model(
                pixel_values=pixel_values_videos,
                image_grid_thw=video_grid_thw,
            ).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeddings)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1) Obtain fused input embeddings
        inputs_embeds = self.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        # 2) Process position_ids through position processor
        position_ids = (
            self.position_id_processor(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            if self.position_id_processor is not None
            else None
        )

        # 3) call super().forward() with concatenated embeddings
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **kwargs,
        )

    def prefill(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        # 1) Obtain fused input embeddings
        inputs_embeds = self.fuse_text_image_video(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        # 2) Process position_ids through position processor
        position_ids = (
            self.position_id_processor(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            if self.position_id_processor is not None
            else None
        )

        # 3) call super().prefill() with concatenated embeddings
        yield from super().prefill(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **kwargs,
        )
