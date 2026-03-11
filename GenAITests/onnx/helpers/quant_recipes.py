# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-ONNX"""

from abc import ABC, abstractmethod
from tqdm import tqdm
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.sequential_mse.seq_mse import SeqMseParams, SequentialMse
from aimet_onnx.experimental.adascale.adascale_optimizer import (
    apply_adascale,
    adascale_model_config_dict,
)

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.generator import Generator
from GenAITests.onnx.models.utils.torch_onnx_interface import kwargs_to_dict


def _get_lm_head_node_names(quantsim: QuantizationSimModel) -> list[str]:
    lm_head_node_names = []
    vocab_size = (
        quantsim.model.model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    )
    node_input_map = {
        node.input[1]: node
        for node in quantsim.model.model.graph.node
        if node.op_type in ("Gemm", "MatMul", "Conv")
    }
    for weight in quantsim.model.model.graph.initializer:
        if vocab_size in weight.dims:
            for suffix in ("", "_updated", "_qdq"):
                candidate_name = weight.name + suffix if suffix else weight.name
                if candidate_name in node_input_map:
                    node = node_input_map[candidate_name]
                    lm_head_node_names.append(node.name)
    return lm_head_node_names


def _prefill_inputs(
    quantsim: QuantizationSimModel,
    generator: Generator,
    dataloader: DataLoader,
    num_iterations: int = None,
) -> list[dict[str, np.ndarray]]:
    input_names = [inp.name for inp in quantsim.session.get_inputs()]
    inputs = []
    if num_iterations is not None:
        dataloader = itertools.islice(dataloader, num_iterations)

    with generator.fp_mode():
        for sample in tqdm(
            dataloader, total=num_iterations, desc="Pre-filling calibration data"
        ):
            inputs.extend(
                list(generator.prefill(sample["input_ids"], sample["attention_mask"]))
            )

    def convert_torch_inputs_to_numpy(
        inputs: tuple[torch.Tensor, ...],
    ) -> dict[str, np.ndarray]:
        return {
            k: v.cpu().detach().numpy()
            for k, v in kwargs_to_dict(input_names, *inputs).items()
        }

    return list(map(convert_torch_inputs_to_numpy, inputs))


class QuantizationTechnique(ABC):
    """Generic AIMET-ONNX GenAI quantization technique"""

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
        # Remove all quantization nodes from the ONNX model.
        quantsim.model.model = quantsim.remove_quantizers(quantsim.model.model)
        quantsim._rebuild_session()


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
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_iterations)

        # Step 2: Calibrate quantization parameters.
        def _forward(session, _):
            for batch in tqdm(inputs, total=len(inputs), desc="Calibrating"):
                session.run(None, batch)

        quantsim.compute_encodings(_forward, tuple())


@YAMLConfigParser.register_recipe
class SeqMSE(QuantizationTechnique):
    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_iterations)

        # Step 2: Optimize weight quantization parameters to minimize layer-wise MSE.
        print("Starting Sequential MSE...")
        params = SeqMseParams(num_batches=num_iterations)
        seq_mse = SequentialMse(
            model=quantsim.model,
            sim=quantsim,
            params=params,
            data_loader=inputs,
            nodes_to_exclude=_get_lm_head_node_names(quantsim),
        )
        seq_mse.apply_seq_mse_algo()

        # Step 3: Calibrate activation quantization parameters.
        def _forward(session, _):
            for batch in tqdm(inputs, total=len(inputs), desc="Calibrating"):
                session.run(None, batch)

        quantsim.compute_encodings(_forward, tuple())


@YAMLConfigParser.register_recipe
class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_batches: int = 32,
        num_iterations: int = 64,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_batches)

        # Step 2: Optimize quantization parameters using AdaScale.
        apply_adascale(
            quantsim,
            inputs,
            adascale_model_config_dict[generator.config.model_type],
            num_iterations,
        )

        # Step 3: Calibrate activation quantization parameters.
        def _forward(session, _):
            for batch in tqdm(inputs, total=len(inputs), desc="Calibrating"):
                session.run(None, batch)

        quantsim.compute_encodings(_forward, tuple())
