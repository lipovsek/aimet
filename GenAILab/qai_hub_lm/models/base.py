# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM base class for GenAI test framework"""

import types
from abc import abstractmethod, ABC
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
from .utils.layer_cache import (
    LayerCacheDescriptor,
    attention_mask_input_names,
    cache_state_names,
    AttentionType,
    _resolve_text_config,
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
        extras: dict[str, torch.nn.Module] = None,
    ):
        self.backbone = backbone
        self.visual = visual
        self.embedding = embedding
        self.config = config
        self.position_id_processor = position_id_processor
        self.extras = extras or {}

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
            text_cfg = _resolve_text_config(llm_config)
            text_cfg.num_hidden_layers = 2
            if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
                text_cfg.layer_types = text_cfg.layer_types[:2]

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
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        """Build ``dynamic_axes`` dict for ``torch.onnx.export``.

        Marks the sequence_length and kv_cache_length dimensions as dynamic so
        that a single ONNX graph can be used with varying sequence lengths.
        """
        axes: dict[str, dict[int, str]] = {
            "input_ids": {1: "sequence_length"},
            "position_ids": {1: "sequence_length"},
            "logits": {1: "sequence_length"},
        } | {
            name: {2: "sequence_length"}
            for name in attention_mask_input_names(layer_cache_descriptors)
        }
        for desc in layer_cache_descriptors:
            i = desc.layer_idx
            if desc.attention_type == AttentionType.LINEAR:
                continue
            axes[f"past_key_{i}_in"] = {2: "kv_cache_length"}
            axes[f"past_value_{i}_in"] = {2: "kv_cache_length"}
        return axes

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
    def get_backbone_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        axes: dict[str, dict[int, str]] = {
            "inputs_embeds": {1: "sequence_length"},
            "attention_mask": {2: "sequence_length"},
            "position_ids": {2: "sequence_length"},
            "logits": {1: "sequence_length"},
        }
        for desc in layer_cache_descriptors:
            i = desc.layer_idx
            if desc.attention_type == AttentionType.LINEAR:
                continue
            axes[f"past_key_{i}_in"] = {2: "kv_cache_length"}
            axes[f"past_value_{i}_in"] = {2: "kv_cache_length"}
        return axes

    @staticmethod
    def get_visual_dynamic_axes(
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
    ) -> dict[str, dict[int, str]]:
        return {}

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
