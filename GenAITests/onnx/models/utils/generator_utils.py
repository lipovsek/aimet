# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for instantiating a Generator object in onnx/test_genai.py"""

import contextlib

from GenAITests.shared.models.base import SimCollection
from GenAITests.shared.models.generator import Generator, VLM_Generator

from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface


def _build_fp_mode(sim_collection: SimCollection):
    """Build a context manager factory that temporarily removes ONNX quantization nodes."""

    @contextlib.contextmanager
    def fp_mode():
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    sim_collection.backbone._remove_quantization_nodes()
                )
                sim_collection.backbone._rebuild_session()
                if sim_collection.visual is not None:
                    stack.enter_context(
                        sim_collection.visual._remove_quantization_nodes()
                    )
                    sim_collection.visual._rebuild_session()
                yield
        finally:
            sim_collection.backbone._rebuild_session()
            if sim_collection.visual is not None:
                sim_collection.visual._rebuild_session()

    return fp_mode


def generator_factory(
    sim_collection: SimCollection,
    generator_cls: type[Generator],
    tokenizer,
    sequence_length,
    context_length,
    visual_output_names=None,
    **model_kwargs,
) -> Generator:
    fp_mode = _build_fp_mode(sim_collection)

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
            fp_mode=fp_mode,
            **model_kwargs,
        )
    else:
        return generator_cls(
            model=TorchONNXInterface(sim_collection.backbone, sim_collection.config),
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            fp_mode=fp_mode,
            **model_kwargs,
        )
