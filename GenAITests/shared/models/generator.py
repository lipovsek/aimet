# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM Generator class to restore HF API on models with static shape constraints"""

from __future__ import annotations

import itertools
import math
from typing import Union

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
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.generation_config = None
        self._config = config

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
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
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

        num_processed_tokens = (
            past_key_values.get_seq_length()
            if not isinstance(past_key_values, tuple)
            else past_key_values[0][1].shape[-2]
        )
        return {
            "input_ids": input_ids[:, num_processed_tokens:],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def slice_inputs_for_inference(
        input_ids: torch.Tensor, attention_mask: torch.Tensor, sequence_length: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        input_length = input_ids.shape[-1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            yield (
                input_ids[:, max(0, idx - sequence_length) : idx],
                attention_mask[:, max(0, idx - sequence_length) : idx],
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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        pad_token: int = 0,
    ) -> tuple[torch.Tensor, ...]:
        """Prepare provided inputs for model forward pass with static graph constraints"""
        device = input_ids.device
        batch_size, input_length = input_ids.shape

        # Create attention mask if one is not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Pad input ids and attention mask to [batch_size, sequence_length] (necessary for static shape requirements)
        input_ids_extension = torch.full(
            (batch_size, sequence_length - input_length),
            fill_value=pad_token,
            dtype=input_ids.dtype,
            device=device,
        )
        padded_input_ids = torch.cat((input_ids_extension, input_ids), dim=-1)
        padded_attention_mask = torch.cat(
            (torch.zeros_like(input_ids_extension), attention_mask), dim=-1
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
        cm_attention_mask = cm_attention_mask.clip(-100, 0)

        # Compute position_ids
        position_ids = torch.cumsum(padded_attention_mask, dim=1, dtype=torch.int32) - 1
        position_ids = position_ids.clip(0, context_length - 1)
        position_ids = position_ids[..., -sequence_length:]

        return (
            padded_input_ids.to(dtype=torch.int32),
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
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: DynamicCache = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Create attention mask if one does not exist
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        global_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]] = {
            "past_key_values": []
            if past_key_values is None or past_key_values.get_seq_length() == 0
            else list(itertools.chain.from_iterable(past_key_values.to_legacy_cache()))
        }

        for input_ids_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_ids, attention_mask, self.sequence_length
        ):
            prepared_inputs = self.prepare_inputs(
                model=self.model,
                input_ids=input_ids_slice,
                attention_mask=attention_mask_slice,
                past_key_values=global_outputs["past_key_values"],
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                pad_token=getattr(self.tokenizer, "eos_token_id", 0),
            )

            local_outputs = self.model(*prepared_inputs)

            self.combine_local_and_global_outputs(
                input_ids_slice.shape[-1], local_outputs, global_outputs
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
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: DynamicCache = None,
        **kwargs,
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        # Create attention mask if one does not exist
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # slice input ids and attention mask to drop last few tokens
        total_num_inferences = math.ceil(input_ids.shape[-1] / self.sequence_length)
        num_tokens_to_preconsume = (total_num_inferences - 1) * self.sequence_length

        input_ids_to_preconsume = input_ids[:, :num_tokens_to_preconsume]
        attention_mask_to_preconsume = attention_mask[:, :num_tokens_to_preconsume]

        preconsumed_outputs: dict[str, Union[torch.Tensor | list[torch.Tensor]]] = {
            "past_key_values": []
            if past_key_values is None or past_key_values.get_seq_length() == 0
            else list(itertools.chain.from_iterable(past_key_values.to_legacy_cache()))
        }

        for input_ids_slice, attention_mask_slice in self.slice_inputs_for_inference(
            input_ids_to_preconsume, attention_mask_to_preconsume, self.sequence_length
        ):
            prepared_inputs = self.prepare_inputs(
                self.model,
                input_ids_slice,
                attention_mask_slice,
                preconsumed_outputs["past_key_values"],
                self.sequence_length,
                self.context_length,
            )

            yield prepared_inputs

            local_outputs = self.model(*prepared_inputs)
            self.combine_local_and_global_outputs(
                input_ids_slice.shape[-1],
                local_outputs,
                preconsumed_outputs,
            )

        remaining_input_ids = input_ids[:, num_tokens_to_preconsume:]
        remaining_attention_mask = attention_mask[:, num_tokens_to_preconsume:]
        prefilled_inputs = self.prepare_inputs(
            self.model,
            remaining_input_ids,
            remaining_attention_mask,
            preconsumed_outputs["past_key_values"],
            self.sequence_length,
            self.context_length,
            getattr(self.tokenizer, "eos_token_id", 0),
        )

        yield prefilled_inputs
