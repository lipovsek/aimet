# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-Torch"""

from abc import ABC, abstractmethod
import itertools
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale
from aimet_torch.experimental.spinquant.spinquant_optimizer import apply_spinquant
from aimet_torch.v2.utils import remove_all_quantizers
from aimet_torch import QuantizationSimModel
from aimet_torch.utils import change_tensor_device_placement
from aimet_torch.v2.seq_mse import apply_seq_mse
from aimet_torch.experimental.omniquant import apply_omniquant

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.generator import Generator


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
            inputs.extend(
                change_tensor_device_placement(
                    list(
                        generator.prefill(
                            sample["input_ids"].to(device=generator.device),
                            sample["attention_mask"].to(device=generator.device),
                        )
                    ),
                    device=device if device else generator.device,
                )
            )

    return inputs


def _compute_encodings(
    quantsim: QuantizationSimModel,
    generator: Generator,
    dataloader: DataLoader,
    num_iterations: int = None,
):
    """Internal helper function to compute encodings on quantsim model"""
    assert quantsim.model == generator.model

    if num_iterations is None:
        num_iterations = len(dataloader)

    def callback(_):
        sliced_dataloader = itertools.islice(dataloader, num_iterations)
        for batch in tqdm(sliced_dataloader, total=num_iterations, desc="Calibrating"):
            generator(input_ids=batch["input_ids"].to(device=generator.device))

    quantsim.compute_encodings(callback)


class QuantizationTechnique(ABC):
    """Generic GenAI quantization technique"""

    @staticmethod
    @abstractmethod
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        """Apply quantization technique"""


@YAMLConfigParser.register_recipe
class RemoveQuantization(QuantizationTechnique):
    """Remove all quantization nodes from quantsim model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        # Remove all quantization wrappers to get an FP baseline.
        remove_all_quantizers(quantsim.model)


@YAMLConfigParser.register_recipe
class Skip(QuantizationTechnique):
    """Do nothing. Useful for testing fully precomputed encodings."""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
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
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        # Calibrate quantization parameters by running data through the model.
        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class SeqMSE(QuantizationTechnique):
    """Apply SeqMSE to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(generator, dataloader, 20, torch.device("cpu"))

        # Step 2: Optimize weight quantization parameters to minimize layer-wise MSE.
        apply_seq_mse(quantsim, inputs, num_candidates=20)

        # Step 3: Calibrate activation quantization parameters.
        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: Dataset,
        num_batches: int = 20,
        num_iterations: int = 1500,
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

        # Step 3: Calibrate activation quantization parameters.
        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class OmniQuant(QuantizationTechnique):
    """Apply OmniQuant to model"""

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_batches: int = 40,
        num_iterations: int = 800,
    ):
        class LimitedBatchDataLoader:
            """Internal helper class to reduce number of accessible batches in Dataloader"""

            def __init__(self, dataloader, num_batches):
                self.dataloader = dataloader
                self.num_batches = num_batches
                self.current_batch = 0

            def __iter__(self):
                # pylint: disable=attribute-defined-outside-init
                self.iterator = iter(self.dataloader)
                self.current_batch = 0
                return self

            def __next__(self):
                if self.current_batch < self.num_batches:
                    self.current_batch += 1
                    return next(self.iterator)
                raise StopIteration

            def __len__(self):
                return min(len(self.dataloader), self.num_batches)

        # Step 1: Apply OmniQuant optimization.
        apply_omniquant(
            quant_sim=quantsim,
            dataloader=LimitedBatchDataLoader(dataloader, num_batches=num_batches),
            forward_fn=lambda model, input: generator(**input),
            num_iterations=num_iterations,
        )

        # Step 2: Calibrate activation quantization parameters.
        _compute_encodings(quantsim, generator, dataloader, num_iterations=40)


@YAMLConfigParser.register_recipe
class SpinQuant(QuantizationTechnique):
    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel, generator: Generator, dataloader: DataLoader
    ):
        # Step 1: Untie embed_tokens and lm_head weights if they are shared.
        if (
            quantsim.model.model.model.embed_tokens.weight
            is quantsim.model.model.lm_head.weight
        ):
            old_weight = quantsim.model.model.lm_head.weight
            new_weight = torch.nn.Parameter(
                old_weight.data.clone().detach().to(old_weight.device),
                requires_grad=True,
            )
            quantsim.model.model.lm_head.weight = new_weight

        # Step 2: Apply SpinQuant rotation.
        apply_spinquant(model=quantsim.model.model)

        # Step 3: Calibrate quantization parameters.
        _compute_encodings(quantsim, generator, dataloader, num_iterations=20)


@YAMLConfigParser.register_recipe
class SpinQuant_AdaScale(QuantizationTechnique):
    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: Dataset,
        num_batches: int = 20,
        num_iterations: int = 1500,
    ):
        # Step 1: Untie embed_tokens and lm_head weights if they are shared.
        if (
            quantsim.model.model.model.embed_tokens.weight
            is quantsim.model.model.lm_head.weight
        ):
            old_weight = quantsim.model.model.lm_head.weight
            new_weight = torch.nn.Parameter(
                old_weight.data.clone().detach().to(old_weight.device),
                requires_grad=True,
            )
            quantsim.model.model.lm_head.weight = new_weight

        # Step 2: Apply SpinQuant rotation.
        apply_spinquant(model=quantsim.model.model)

        # Step 3: Apply AdaScale optimization.
        AdaScale.apply(quantsim, generator, dataloader, num_batches, num_iterations)
