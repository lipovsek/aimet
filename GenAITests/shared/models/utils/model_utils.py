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
            k = new_past_key_values.key_cache[layer]
            v = new_past_key_values.value_cache[layer]
            flat_output_past_key_values += [k, v]

        return lm_logits, *flat_output_past_key_values
