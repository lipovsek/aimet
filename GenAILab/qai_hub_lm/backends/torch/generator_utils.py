# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Torch generator utils"""

import contextlib
import dataclasses
import torch

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.utils import place_model
from aimet_torch.v2.utils import remove_all_quantizers

from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator


@contextlib.contextmanager
def place_collection(models: SimCollection, device: torch.device):
    """
    Temporarily place all non-None models in the collection on the specified device.

    :param models: SimCollection containing QuantSim models
    :param device: Target device to place models on
    """
    with contextlib.ExitStack() as stack:
        for field in dataclasses.fields(models):
            sim = getattr(models, field.name)
            if (
                sim is not None
                and isinstance(sim, QuantizationSimModel)
                and isinstance(sim.model, torch.nn.Module)
            ):
                stack.enter_context(place_model(sim.model, device))
            elif sim is not None and isinstance(sim, torch.nn.Module):
                stack.enter_context(place_model(sim, device))
        yield


def _build_fp_mode(sim_collection: SimCollection):
    """Build a context manager factory that temporarily disables all quantizers."""

    @contextlib.contextmanager
    def fp_mode():
        with contextlib.ExitStack() as stack:
            stack.enter_context(remove_all_quantizers(sim_collection.backbone.model))
            if sim_collection.visual is not None:
                stack.enter_context(remove_all_quantizers(sim_collection.visual.model))
            if sim_collection.embedding is not None:
                stack.enter_context(remove_all_quantizers(sim_collection.embedding))
            yield

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
            backbone_model=sim_collection.backbone.model,
            vision_model=sim_collection.visual.model,
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
            model=sim_collection.backbone.model,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            fp_mode=fp_mode,
            **model_kwargs,
        )
