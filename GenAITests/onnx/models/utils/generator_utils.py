# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for instantiating a Generator object in onnx/test_genai.py"""

from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.generator import Generator, VLM_Generator

from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface


def generator_factory(
    sim_collection: SimCollection,
    generator_cls: type[Generator],
    tokenizer,
    sequence_length,
    context_length,
    visual_output_names=None,
    **model_kwargs,
) -> Generator:
    if sim_collection.is_vlm():
        assert issubclass(generator_cls, VLM_Generator)
        return generator_cls(
            backbone_model=TorchONNXInterface(
                sim_collection.backbone, sim_collection.config.text_config
            ),
            vision_model=TorchONNXInterface(
                sim_collection.visual, sim_collection.config
            ),
            embedding=sim_collection.embedding,
            tokenizer=tokenizer,
            position_id_processor=sim_collection.position_id_processor,
            sequence_length=sequence_length,
            context_length=context_length,
            config=sim_collection.config,
            visual_output_names=visual_output_names,
            **model_kwargs,
        )
    else:
        return generator_cls(
            model=TorchONNXInterface(sim_collection.backbone, sim_collection.config),
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            **model_kwargs,
        )
