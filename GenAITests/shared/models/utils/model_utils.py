# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for building GenAI models"""

import torch
from transformers import PreTrainedModel, DynamicCache


class ONNXExportableModuleWithCache(torch.nn.Module):
    """
    Helper class to enable Torch JIT trace and ONNX export of HuggingFace models that produce and consume Cache objects
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    @property
    def device(self):
        """Return model device"""
        return self.model.device

    @property
    def dtype(self):
        """Return model dtype"""
        return self.model.dtype

    @property
    def config(self):
        """Return model config"""
        return self.model.config

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        *past_key_values: torch.Tensor,
    ):
        """Redefine model forward to convert to/from Huggingface DynamicCache objects"""
        kv_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(
            zip(past_key_values[::2], past_key_values[1::2])
        ):
            kv_cache.update(k, v, layer_idx, {})

        lm_logits, new_past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            num_logits_to_return=0,
            return_dict=False,
        )

        flat_output_past_key_values = []
        for layer in range(len(new_past_key_values)):
            if hasattr(new_past_key_values, "value_cache"):
                keys = new_past_key_values.key_cache[layer]
                values = new_past_key_values.value_cache[layer]
            elif hasattr(new_past_key_values.layers[layer], "keys"):
                keys = new_past_key_values.layers[layer].keys
                values = new_past_key_values.layers[layer].values
            else:
                keys = new_past_key_values.layers[layer][0]
                values = new_past_key_values.layers[layer][1]
            flat_output_past_key_values += [keys, values]

        return lm_logits, *flat_output_past_key_values


class ONNXExportableBackboneWithCache(ONNXExportableModuleWithCache):
    def __init__(self, model: PreTrainedModel, lm_head: torch.nn.Module):
        super().__init__(model)
        self.lm_head = lm_head

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        *past_key_values: torch.Tensor,
    ):
        """Redefine model forward to convert to/from Huggingface DynamicCache objects"""
        kv_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(
            zip(past_key_values[::2], past_key_values[1::2])
        ):
            kv_cache.update(k, v, layer_idx, {})

        hidden_states, new_past_key_values = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            num_logits_to_return=0,
            inputs_embeds=inputs_embeds,
            return_dict=False,
        )
        lm_logits = self.lm_head(hidden_states)

        flat_output_past_key_values = []
        for layer in range(len(new_past_key_values)):
            if hasattr(past_key_values, "value_cache"):
                keys = new_past_key_values.key_cache[layer]
                values = new_past_key_values.value_cache[layer]
            elif hasattr(new_past_key_values.layers[layer], "keys"):
                keys = new_past_key_values.layers[layer].keys
                values = new_past_key_values.layers[layer].values
            else:
                keys = new_past_key_values.layers[layer][0]
                values = new_past_key_values.layers[layer][1]
            flat_output_past_key_values += [keys, values]

        return lm_logits, *flat_output_past_key_values
