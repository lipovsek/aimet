# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM base class for GenAI test framework"""

import itertools
import types
from abc import abstractmethod, ABC
from pathlib import Path
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from .generator import Generator, VLM_Generator


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
    @abstractmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        """Instantiate model"""
        pass

    @staticmethod
    @abstractmethod
    def instantiate_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
        """Instantiate model tokenizer"""

    @classmethod
    @abstractmethod
    def instantiate_quantsim(cls, *args, **kwargs) -> SimCollection:
        """Instantiate QuantSim models for components"""
        pass

    @classmethod
    @abstractmethod
    def get_sample_backbone_inputs(
        cls, model, context_length: int, sequence_length: int
    ) -> tuple[torch.Tensor, ...]:
        """Get sample inputs for LLM backbone QuantSim instantiation or ONNX export"""
        pass

    @staticmethod
    def get_quantsim_config() -> str:
        """Get default QuantSim config"""
        config_path = Path(__file__).parent / "config/default_config.json"
        return str(config_path.resolve())

    @staticmethod
    def get_backbone_input_names(num_layers: int) -> tuple[str, ...]:
        """Get input names for the backbone model"""
        names = ["input_ids", "attention_mask", "position_ids"]
        kv_names = zip(
            [f"past_key_{i}_in" for i in range(num_layers)],
            [f"past_value_{i}_in" for i in range(num_layers)],
        )
        return tuple(names + list(itertools.chain.from_iterable(kv_names)))

    @staticmethod
    def get_backbone_output_names(num_layers: int) -> tuple[str, ...]:
        """Get output names for the backbone model"""
        names = ["logits"]
        kv_names = zip(
            [f"past_key_{i}_out" for i in range(num_layers)],
            [f"past_value_{i}_out" for i in range(num_layers)],
        )
        return tuple(names + list(itertools.chain.from_iterable(kv_names)))

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
        cls, config: PretrainedConfig
    ) -> tuple[torch.Tensor, ...]:
        """Get sample inputs for visual model QuantSim instantiation or ONNX export"""
        pass

    @staticmethod
    def get_backbone_input_names(num_layers: int) -> tuple[str, ...]:
        """Get input names for the backbone model"""
        names = ["inputs_embeds", "attention_mask", "position_ids"]
        kv_names = zip(
            [f"past_key_{i}_in" for i in range(num_layers)],
            [f"past_value_{i}_in" for i in range(num_layers)],
        )
        return tuple(names + list(itertools.chain.from_iterable(kv_names)))

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
