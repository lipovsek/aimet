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
from aimet_onnx.experimental.spinquant import apply_spinquant

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import kwargs_to_dict


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

    def _to_numpy(tensors: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
        return {
            k: v.cpu().detach().numpy()
            for k, v in kwargs_to_dict(input_names, *tensors).items()
        }

    with generator.fp_mode():
        for sample in tqdm(
            dataloader, total=num_iterations, desc="Pre-filling calibration data"
        ):
            sample_kwargs = {
                k: v.to(device=generator.device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            # Convert to CPU numpy immediately to free CUDA memory before
            # fp_mode exits and rebuilds the full quantized session.
            inputs.extend([_to_numpy(t) for t in generator.prefill(**sample_kwargs)])

    return inputs


class QuantizationTechnique(ABC):
    """Generic AIMET-ONNX GenAI quantization technique"""

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
        # Remove all quantization nodes from the ONNX model.
        quantsim.model.model = quantsim.remove_quantizers(quantsim.model.model)
        quantsim._rebuild_session()


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
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
        **kwargs,
    ):
        def _forward(_):
            sliced_dataloader = itertools.islice(dataloader, num_iterations)
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

        quantsim.compute_encodings(_forward)


@YAMLConfigParser.register_recipe
class SeqMSE(QuantizationTechnique):
    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_iterations)

        # Step 2: Pre-compute param encodings with a real input so that
        # SequentialMse's internal _compute_param_encodings(overwrite=False)
        # finds them already initialized and skips make_dummy_input, which
        # generates random values that break structured inputs like
        # image_grid_thw in vision models.
        quantsim._compute_param_encodings(dummy_input=inputs[0], overwrite=False)

        # Step 3: Optimize weight quantization parameters to minimize layer-wise MSE.
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


@YAMLConfigParser.register_recipe
class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_batches: int = 32,
        num_iterations: int = 64,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_batches)

        # Step 2: Pre-compute param encodings with a real input (same
        # reason as SeqMSE — avoids make_dummy_input for vision models).
        quantsim._compute_param_encodings(dummy_input=inputs[0], overwrite=False)

        # Step 3: Optimize quantization parameters using AdaScale.
        apply_adascale(
            quantsim,
            inputs,
            adascale_model_config_dict[generator.config.model_type],
            num_iterations,
        )


@YAMLConfigParser.register_recipe
class SpinQuant(QuantizationTechnique):
    """Apply SpinQuant: R1 rotation to model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        component: str = "backbone",
        **kwargs,
    ):
        if component == "backbone":
            if isinstance(generator, VLM_Generator):
                apply_spinquant(
                    backbone_sim=quantsim,
                    visual_sim=generator.vision_model.quantsim,
                    embedding=generator.embedding.weight,
                )
            else:
                apply_spinquant(quantsim)
        elif component == "visual":
            print(
                "WARNING: SpinQuant is a no-op on visual — rotation was already applied "
                "to merger_linear2 when SpinQuant ran on the backbone."
            )
            return
