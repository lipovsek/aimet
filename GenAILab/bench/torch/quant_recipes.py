# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-Torch"""

from abc import ABC, abstractmethod
import itertools
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from aimet_torch import QuantizationSimModel
from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale
from aimet_torch.experimental.spinquant.spinquant_optimizer import (
    SpinQuant as SpinQuantOptimizer,
)
from aimet_torch.v2.seq_mse import apply_seq_mse
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.v2.utils import remove_all_quantizers
from aimet_torch.utils import change_tensor_device_placement

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator


def _prefill_inputs(
    generator: Generator,
    dataloader: DataLoader,
    num_iterations: int = None,
    device: torch.device = None,
):
    inputs = []
    if num_iterations is not None:
        dataloader = itertools.islice(dataloader, num_iterations)

    with generator.fp_mode(), torch.no_grad():
        for sample in tqdm(
            dataloader,
            total=num_iterations if num_iterations else len(dataloader),
            desc="Pre-filling calibration data",
        ):
            sample_kwargs = {
                k: v.to(device=generator.device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            inputs.extend(
                change_tensor_device_placement(
                    [tuple(d.values()) for d in generator.prefill(**sample_kwargs)],
                    device=device if device else generator.device,
                )
            )

    return inputs


class QuantizationTechnique(ABC):
    """Generic GenAI quantization technique"""

    @classmethod
    def cacheable(cls):
        return False

    @staticmethod
    @abstractmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **kwargs,
    ):
        """Apply quantization technique"""


@YAMLConfigParser.register_recipe
class RemoveQuantization(QuantizationTechnique):
    """Remove all quantization nodes from quantsim model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **kwargs,
    ):
        # Remove all quantization wrappers to get an FP baseline.
        remove_all_quantizers(quantsim.model)


@YAMLConfigParser.register_recipe
class Skip(QuantizationTechnique):
    """Do nothing. Useful for testing fully precomputed encodings."""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **kwargs,
    ):
        pass


@YAMLConfigParser.register_recipe
class Calibration(QuantizationTechnique):
    """Calibrate quantization parameters.

    Granularity (PCQ/LPBQ/BQ) is configured via the ``precision:`` section.
    This recipe simply runs calibration.
    """

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
        **kwargs,
    ):
        # Calibrate quantization parameters by running data through the model.
        if num_iterations is None:
            num_iterations = len(dataloader)
        sliced_dataloader = itertools.islice(dataloader, num_iterations)

        with compute_encodings(quantsim.model), torch.no_grad():
            for batch in tqdm(
                sliced_dataloader, total=num_iterations, desc="Calibrating"
            ):
                inputs = {
                    k: v.to(device=generator.device)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in batch.items()
                }
                generator(**inputs)


@YAMLConfigParser.register_recipe
class SeqMSE(QuantizationTechnique):
    """Apply SeqMSE to model"""

    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(generator, dataloader, 20, torch.device("cpu"))

        # Step 2: Optimize weight quantization parameters to minimize layer-wise MSE.
        apply_seq_mse(quantsim, inputs, num_candidates=20)


@YAMLConfigParser.register_recipe
class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: Dataset,
        num_batches: int = 20,
        num_iterations: int = 1500,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(
            generator, dataloader, num_batches, torch.device("cpu")
        )

        # Step 2: Optimize quantization parameters using AdaScale.
        apply_adascale(
            quantsim,
            inputs,
            num_iterations=num_iterations,
        )


@YAMLConfigParser.register_recipe
class SpinQuant(QuantizationTechnique):
    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        component: str = "backbone",
        **kwargs,
    ):
        if component == "backbone":
            SpinQuant._apply_to_backbone(quantsim, generator)
        elif component == "visual":
            SpinQuant._apply_to_visual(quantsim, generator)

    @staticmethod
    def _apply_to_backbone(quantsim: QuantizationSimModel, generator: Generator):
        wrapper = quantsim.model
        # Resolve decoder model and lm_head based on wrapper structure.
        # VLMs: wrapper.model = language_model, wrapper.lm_head = lm_head
        # LLMs: wrapper.model = full HF model (has .model and .lm_head)
        if hasattr(wrapper, "lm_head") and wrapper.lm_head:
            decoder_model = wrapper.model
            lm_head = wrapper.lm_head
        else:
            decoder_model = wrapper.model.model
            lm_head = wrapper.model.lm_head

        # Untie embed_tokens and lm_head weights if they are shared.
        if decoder_model.embed_tokens.weight is lm_head.weight:
            old_weight = lm_head.weight
            lm_head.weight = torch.nn.Parameter(
                old_weight.data.clone().detach().to(old_weight.device),
                requires_grad=True,
            )

        SpinQuantOptimizer._apply_spinquant_to_decoder_stack(decoder_model, lm_head)

        # Apply SpinQuant rotation to the embedding table if available and separate from the embedding table in decoder_model.
        # There is technically an "embed_tokens" in the backbone model (that is rotated by _apply_spinquant_to_decoder_stack)
        # but we have our generator object hold a separate copy of this, so we need to rotate independently
        if (
            isinstance(generator, VLM_Generator)
            and generator.embedding is not decoder_model.embed_tokens
        ):
            hidden_size = decoder_model.embed_tokens.weight.shape[-1]
            generator.embedding = SpinQuantOptimizer._apply_spinquant_to_embedding(
                generator.embedding, hidden_size
            )

    @staticmethod
    def _apply_to_visual(quantsim: QuantizationSimModel, generator: Generator):
        visual_encoder = quantsim.model.visual
        # Determine language backbone hidden size from the generator's config. Could also be determined from weight shapes
        config = generator.config
        text_config = config.text_config if hasattr(config, "text_config") else config
        language_backbone_hidden_size = text_config.hidden_size

        # Find mergers and interface from SpinQuant model config registry.
        vlm_config = SpinQuantOptimizer._find_vlm_config(quantsim.model)
        merger_modules = (
            [
                m
                for m in quantsim.model.modules()
                if isinstance(m, vlm_config.merger_type)
            ]
            if vlm_config
            else None
        )
        merger_interface = vlm_config.merger_interface if vlm_config else None

        SpinQuantOptimizer._apply_spinquant_to_visual_encoder(
            visual_encoder=visual_encoder,
            language_backbone_hidden_size=language_backbone_hidden_size,
            merger_modules=merger_modules,
            merger_interface=merger_interface,
        )
