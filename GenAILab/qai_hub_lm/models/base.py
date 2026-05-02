# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM base class for GenAI test framework"""

import types
from abc import abstractmethod, ABC
from pathlib import Path
import torch
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.cache_utils import DynamicCache

from .generator import Generator, VLM_Generator
from GenAILab.qai_hub_lm.utils.layer_cache import (
    LayerCacheDescriptor,
    attention_mask_input_names,
    cache_state_names,
)


@dataclass
class SimCollection:
    """Dataclass to hold QuantSim models for different parts of the LLM"""

    backbone: "QuantizationSimModel"
    visual: "QuantizationSimModel"
    embedding: torch.nn.Module
    config: PretrainedConfig

    def __init__(
        self,
        backbone: "QuantizationSimModel",
        visual: "QuantizationSimModel" = None,
        embedding: torch.nn.Module = None,
        config: PretrainedConfig = None,
        position_id_processor: types.FunctionType = None,
    ):
        self.backbone = backbone
        self.visual = visual
        self.embedding = embedding
        self.config = config
        self.position_id_processor = position_id_processor

    def is_vlm(self) -> bool:
        return self.visual is not None


class LLM(ABC):
    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        """Instantiate model"""
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

        return AutoModelForCausalLM.from_pretrained(model_id, config=llm_config)

    @staticmethod
    def instantiate_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
        """Instantiate model tokenizer"""
        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    @abstractmethod
    def instantiate_quantsim(cls, *args, **kwargs) -> SimCollection:
        """Instantiate QuantSim models for components"""
        pass

    @classmethod
    @abstractmethod
    def get_sample_backbone_inputs(
        cls,
        model,
        context_length: int,
        sequence_length: int,
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Get sample inputs for LLM backbone QuantSim instantiation or ONNX export"""
        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        prepared = Generator.prepare_inputs(
            model=model,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            layer_cache_descriptors=layer_cache_descriptors,
        )
        return tuple(prepared.values())

    @staticmethod
    def get_quantsim_config() -> str:
        """Get default QuantSim config"""
        config_path = Path(__file__).parent / "config/default_config.json"
        return str(config_path.resolve())

    @staticmethod
    def get_cache_type() -> type:
        """
        Returns ``DynamicCache`` by default. Models with hybrid attention
        (e.g. linear + full) can override this to return ``HybridCache``.
        """
        return DynamicCache

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> tuple[str, ...]:
        """Get input names for the backbone model."""
        return tuple(
            ["input_ids"]
            + attention_mask_input_names(layer_cache_descriptors)
            + ["position_ids"]
            + cache_state_names(layer_cache_descriptors, "in")
        )

    @staticmethod
    def get_backbone_output_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> tuple[str, ...]:
        """Get output names for the backbone model."""
        return tuple(["logits"] + cache_state_names(layer_cache_descriptors, "out"))

    @staticmethod
    def use_dynamo_export() -> bool:
        """Whether to use dynamo-based ONNX export. Models with ops unsupported
        by the TorchScript tracer (e.g. data-dependent control flow) should
        override this to return True."""
        return False

    @staticmethod
    def get_generator_cls() -> type[Generator]:
        return Generator


class VLM(LLM):
    @classmethod
    @abstractmethod
    def instantiate_position_processor(cls):
        pass

    @classmethod
    @abstractmethod
    def get_sample_vision_inputs(
        cls,
        config: PretrainedConfig,
        image_size: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Get sample inputs for visual model QuantSim instantiation or ONNX export"""
        pass

    @staticmethod
    def get_backbone_input_names(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> tuple[str, ...]:
        """Get input names for the backbone model."""
        return tuple(
            ["inputs_embeds"]
            + attention_mask_input_names(layer_cache_descriptors)
            + ["position_ids"]
            + cache_state_names(layer_cache_descriptors, "in")
        )

    @staticmethod
    @abstractmethod
    def get_visual_input_names() -> tuple[str, ...]:
        """Get input names for the visual model"""
        pass

    @staticmethod
    @abstractmethod
    def get_visual_output_names() -> tuple[str, ...]:
        """Get output names for the visual model"""
        pass

    @staticmethod
    def get_generator_cls() -> type[VLM_Generator]:
        return VLM_Generator
