# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-2.5 model class"""

import torch

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.qwen2 import modeling_qwen2

from GenAITests.shared.models.base import LLM
from GenAITests.shared.models.generator import Generator


class Qwen_25(LLM):
    """Generic quantized Qwen 2"""

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def instantiate_model(cls, model_id: str, small_model=False) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2
            if (
                hasattr(llm_config, "layer_types")
                and llm_config.layer_types is not None
            ):
                llm_config.layer_types = llm_config.layer_types[:2]
        return modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> PreTrainedTokenizer:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID
        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def get_sample_backbone_inputs(cls, model, context_length, sequence_length):
        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = Generator.prepare_inputs(
            model=model,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
        )
        return assembled_dummy_inputs
