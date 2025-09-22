# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AdaScale implementation"""

from typing import Collection, Dict

import copy
from dataclasses import dataclass
from typing import Type
import numpy as np
import torch

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from aimet_common.utils import AimetLogger  # pylint: disable=import-error
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
    get_conv_linear_layers_decoder_block,
)
from aimet_onnx.experimental.adascale.activation_sampler import ActivationSampler

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


_QT_SAMPLING_PROB = 0.5
_LOSS_FN = torch.nn.MSELoss()


@dataclass
class AdaScaleModelConfig:
    block_type: Type = None  # block types to use in a given model
    beta_gamma_lr: float = 1e-3  # lr for beta and gamma
    scales_lr: float = 5e-4  # lr for s2, s3, [s4]


# mapping of model type and the corresponding adascale config
adascale_model_config_dict = {
    "LlamaModel": AdaScaleModelConfig(
        block_type=LlamaDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "Qwen2Model": AdaScaleModelConfig(
        block_type=Qwen2DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "MistralModel": AdaScaleModelConfig(
        block_type=MistralDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
}


class AdaScale:
    """
    AdaScale is PTQ technique which performs Knowledge Distillation on blocks of modules by using the FP32 output as its
    reference output. Adascale is based on FlexRound: https://arxiv.org/abs/2306.00317 but integrates LWC from Omniquant.

    The optimization is performed on a block-by-block basis by comparing the quantized output of the block with its FP32
    equivalent and by training the parameters (gamma, beta, s2, s3) which are temporarily introduced in every supported
    module.

    A block is defined as a non-leaf module which takes in one activation input tensor and outputs one activation tensor
    Currently only Linear layers are supported, and all the linears in a block are optimized at the same time.

    While performing the optimization, the activation quantizers are disabled, linear modules' weight quantizers are
    changed to specialized QDQ (with learnable parameters introduced) and rest of the param's are left quantized with
    default QuantizeDequantize.
    """

    # pylint: disable=unused-argument, unused-variable

    @classmethod
    def apply_adascale(
        cls,
        qsim: QuantizationSimModel,
        inputs: Collection[Dict[str, np.ndarray]],
        adascale_model_config: AdaScaleModelConfig,
        num_iterations: int = 1500,
    ):
        """
        :param qsim: Quantization Sim model
        :param inputs: (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization.
        :param adascale_model_config: Adascale model config. There are pre-defined configs for
                                      LlamaModel, Qwen2Model, MistralModel. For other models use AdaScaleModelConfig
        :param num_iterations: Number of iterations to optimize for during AdaScale

        Example usage:
            >>> model = DummyModel()
            >>> inputs = ...
            >>> adascale_model_config = adascale_model_config['LlamaModel']
            >>> sim = QuantizationSimModel(model)
            >>> apply_adascale(sim, inputs, adascale_model_config, num_iterations=num_iterations)
            >>> sim.compute_encodings(...)
            >>> sim.export(...)

        .. note::
        1. apply_adascale modifies the weights in-place in the model
        2. compute encodings should not be called before the apply_adascale call
        3. Activation quantizers will remain uninitialized throughout the feature, and so compute encodings needs to be called by the user afterwards. This is so activation encodings will be computed with updated weights taken into account.

        Warning: This feature is currently considered experimental pending API changes
        """
        # pylint: disable=protected-access
        qsim._compute_param_encodings(overwrite=False)

        adascale_blocks_end_points, modules = cls._get_blocks(qsim)

        device = int(
            qsim.session.get_provider_options()
            .get("CUDAExecutionProvider", {})
            .get("device_id", "0")
        )

        fp32_model = copy.deepcopy(qsim.model.model)
        fp32_model = QuantizationSimModel.remove_quantizers(fp32_model)

        for start_layer, _ in adascale_blocks_end_points:
            fp_input, qsim_input = cls._sample_block_inputs(
                qsim, fp32_model, start_layer.outputs[0], inputs, device
            )
            print(fp_input, qsim_input)

    @staticmethod
    def _get_blocks(qsim):
        """helper to get all the blocks in the model represented by adascale_model_config_dict"""
        end_points = get_decoder_blocks_end_points(qsim)
        conv_linear_modules = get_conv_linear_layers_decoder_block(qsim, end_points)
        return end_points, conv_linear_modules

    @staticmethod
    def _copy_weights_onnx_to_pt(block_type, modules):
        raise NotImplementedError()

    @staticmethod
    def _sample_block_inputs(qsim, fp32_model, activation, inputs, device):
        qsim_sess = ActivationSampler(activation.name, qsim.model.model, device)
        qsim_input = qsim_sess.sample_acts(inputs)
        fp32_sess = ActivationSampler(activation.name, fp32_model, device)

        fp_input = fp32_sess.sample_acts(inputs)
        return fp_input, qsim_input

    @staticmethod
    def _copy_weights_encodings_pt_to_onnx(pt_block):
        raise NotImplementedError()


apply_adascale = AdaScale.apply_adascale
