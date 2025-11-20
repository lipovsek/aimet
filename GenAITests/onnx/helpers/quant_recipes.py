# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-ONNX"""

from abc import ABC, abstractmethod
from tqdm import tqdm
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from aimet_onnx.quantsim import (
    QuantizationSimModel,
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_onnx.sequential_mse.seq_mse import SeqMseParams, SequentialMse

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.generator import Generator
from GenAITests.onnx.models.utils.torch_onnx_interface import kwargs_to_dict


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

    with quantsim._remove_quantization_nodes():
        quantsim._rebuild_session()
        for sample in tqdm(
            dataloader, total=num_iterations, desc="Pre-filling calibration data"
        ):
            inputs.extend(
                list(generator.prefill(sample["input_ids"], sample["attention_mask"]))
            )
    quantsim._rebuild_session()

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
        quantsim.model.model = quantsim.remove_quantizers(quantsim.model.model)
        quantsim._rebuild_session()


@YAMLConfigParser.register_recipe
class PCQ(QuantizationTechnique):
    """Apply vanilla PCQ to model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
    ):
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_iterations)

        def _forward(session, _):
            for batch in tqdm(inputs, total=len(inputs), desc="Calibrating"):
                session.run(None, batch)

        quantsim.compute_encodings(_forward, tuple())


@YAMLConfigParser.register_recipe
class LPBQ(QuantizationTechnique):
    """Apply LPBQ to model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
    ):
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_iterations)

        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            op_types=("Gemm", "MatMul", "Conv"),
            bitwidth=4,
            decompressed_bw=8,
            block_size=64,
        )

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
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_iterations)

        print("Starting Sequential MSE...")
        params = SeqMseParams(num_batches=num_iterations)
        seq_mse = SequentialMse(quantsim.model, quantsim, params, inputs)
        seq_mse.apply_seq_mse_algo()

        def _forward(session, _):
            for batch in tqdm(inputs, total=len(inputs), desc="Calibrating"):
                session.run(None, batch)

        quantsim.compute_encodings(_forward, tuple())


@YAMLConfigParser.register_recipe
class LPBQ_SeqMSE(QuantizationTechnique):
    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
    ):
        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            op_types=("Gemm", "MatMul", "Conv"),
            bitwidth=4,
            decompressed_bw=8,
            block_size=64,
        )
        SeqMSE.apply(quantsim, generator, dataloader, num_iterations)
