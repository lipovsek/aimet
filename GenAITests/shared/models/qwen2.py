# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-2.5 model class"""

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.qwen2 import modeling_qwen2

from .base import LLM


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
