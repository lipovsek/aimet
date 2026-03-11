# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""AdaScale implementation"""

import functools
from copy import deepcopy
from dataclasses import dataclass
from types import NoneType
from typing import Callable, List, Any, Tuple, Type, Sequence, Optional, Dict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3Model, Phi3DecoderLayer
from transformers.models.mistral.modeling_mistral import (
    MistralModel,
    MistralDecoderLayer,
)

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
except ImportError:
    Qwen3Model = Qwen3DecoderLayer = None

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLTextModel,
        Qwen2_5_VLDecoderLayer,
    )
except ImportError:
    Qwen2_5_VLTextModel = Qwen2_5_VLDecoderLayer = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextModel,
        Qwen3VLTextDecoderLayer,
    )
except ImportError:
    Qwen3VLTextModel = Qwen3VLTextDecoderLayer = None


from aimet_torch.common.utils import AimetLogger
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn import QuantizedLinear, compute_param_encodings, QuantizedConv2d
from aimet_torch.v2.utils import (
    default_forward_fn,
    remove_all_quantizers,
    remove_activation_quantizers,
)
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.experimental.adascale.adascale_quantizer import (
    AdaScaleQuantizeDequantize,
    AdaScaleLinearQuantizeDequantize,
    AdaScaleConv2dQuantizeDequantize,
)
from aimet_torch.blockwise_sampler import (
    BlockwiseSampler,
    change_tensor_and_cache_device_placement,
    CachedIterable,
)
from aimet_torch.utils import get_device


@dataclass
class AdaScaleModelConfig:
    block_type: Type = None  # block types to use in a given model
    beta_gamma_lr: float = 1e-3  # lr for beta and gamma
    scales_lr: float = 5e-4  # lr for s2, s3, [s4]
    enable_caching_after_block: int = 0
    # for models with ops in between blocks (ex: Qwen3VL), intermediate activations cannot be cached accurately.
    # This param can be used to disable caching for initial blocks until the caching strategy can be used.


# mapping of model type and the corresponding adascale config
adascale_model_config_dict = {
    LlamaModel: AdaScaleModelConfig(
        block_type=LlamaDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    Qwen2Model: AdaScaleModelConfig(
        block_type=Qwen2DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    MistralModel: AdaScaleModelConfig(
        block_type=MistralDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    Phi3Model: AdaScaleModelConfig(
        block_type=Phi3DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
}

if Qwen2_5_VLTextModel is not None and Qwen2_5_VLDecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Qwen2_5_VLTextModel: AdaScaleModelConfig(
                block_type=Qwen2_5_VLDecoderLayer,
                beta_gamma_lr=1e-3,
                scales_lr=5e-4,
            )
        }
    )

if Qwen3Model is not None and Qwen3DecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Qwen3Model: AdaScaleModelConfig(
                block_type=Qwen3DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
            )
        }
    )
if Qwen3VLTextModel is not None and Qwen3VLTextDecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Qwen3VLTextModel: AdaScaleModelConfig(
                block_type=Qwen3VLTextDecoderLayer,
                beta_gamma_lr=1e-3,
                scales_lr=5e-4,
                enable_caching_after_block=3,
            )
        }
    )


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


_QT_SAMPLING_PROB = 0.5
_LOSS_FN = torch.nn.MSELoss()

supported_modules: List = [QuantizedLinear, QuantizedConv2d]


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

    @classmethod
    def apply_adascale(
        cls,
        qsim: QuantizationSimModel,
        data_loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Any], Any] = None,
        num_iterations: int = 1500,
    ):
        """
        :param qsim: Quantization Sim model
        :param data_loader: DataLoader object to load the input data
        :param forward_fn: forward function to run the forward pass of the model
        :param num_iterations: Number of iterations to optimize for during AdaScale BKD

        Note that the forward_fn should take exactly two arguments -
        1) the model
        2) The object returned from the dataloader irrespective of whether it's a tensor/tuple of tensors/dict/etc

        The forward_fn should prepare the "input sample" as needed and call the forward pass in the very end. The forward_fn
        should not be running any sort of eval, creating full dataloader inside the method, etc.

        Example usage:
            >>> model = DummyModel()
            >>> dummy_input = ...
            >>> data_set = DataSet(dummy_input)
            >>> data_loader = DataLoader(data_set, ...)
            >>> sim = QuantizationSimModel(model, dummy_input)
            >>> apply_adascale(sim, data_loader, forward_fn=forward_fn, num_iterations=1500)
            >>> sim.compute_encodings(...)
            >>> sim.export(...)

        .. note::
        1. apply_adascale modifies the weights in-place in the model
        2. compute encodings should not be called before the apply_adascale call
        3. Activation quantizers will remain uninitialized throughout the feature, and so compute encodings needs to be called by the user afterwards. This is so activation encodings will be computed with updated weights taken into account.

        Warning: This feature is currently considered experimental pending API changes
        """
        # pylint: disable=too-many-locals
        if not forward_fn:
            forward_fn = default_forward_fn

        compute_param_encodings(qsim.model)

        adascale_blocks = cls._get_blocks(qsim)

        # replace with adascale weight quantizer which introduces trainable params - beta, gamma, s2, s3
        device = get_device(qsim.model)
        dtype = next(qsim.model.parameters()).dtype

        qsim.model.to(device=torch.device("cpu"), dtype=dtype)

        sampler = BlockwiseSampler(
            qsim,
            adascale_blocks,
            data_loader,
            forward_fn,
            keep_unused_blocks_on_cpu=True,
            cache_activations_on_disk=True,
            disable_caching_until_block=cls._model_specific_blocks_to_disable_caching(
                qsim
            ),
        )

        qsim.model.requires_grad_(False)
        beta_gamma_lr, scales_lr = AdaScale._model_specific_lr(qsim)

        with remove_activation_quantizers(qsim.model):
            for block, fp_block_inputs, qt_block_inputs in sampler.sample(
                device=device, desc="AdaScale blocks processed"
            ):
                fp_block_inputs = CachedIterable(fp_block_inputs)
                qt_block_inputs = _SamplingIterable(
                    CachedIterable(qt_block_inputs),
                    fp_block_inputs,
                    ratio=_QT_SAMPLING_PROB,
                    device=device,
                )
                cls.adascale_block(
                    block,
                    fp_block_inputs,
                    qt_inputs=qt_block_inputs,
                    num_iterations=num_iterations,
                    beta_gamma_lr=beta_gamma_lr,
                    scales_lr=scales_lr,
                )

        del sampler

        cls.fold_adascale_quantizers(qsim.model)

        qsim.model.to(device=device, dtype=dtype)

    @classmethod
    def adascale_block(
        cls,
        block: torch.nn.Module,
        fp_inputs: Sequence[Tuple[Tuple, Dict]],
        qt_inputs: Sequence[Tuple[Tuple, Dict]],
        *,
        num_iterations: int = 1500,
        beta_gamma_lr: float = 1e-3,
        scales_lr: float = 5e-4,
    ):
        """
        Performs AdaScale algorithm to optimize weight quantization of input block to minimize the output MSE
        between floating point and quantized execution of the block.

        Args:
            block: Quantized torch.nn.Module object to optimize
            fp_inputs: Sequence of (args, kwargs) observed by the block during floating-point model execution
            qt_inputs: Sequence of (args, kwargs) observed by the block during quantized model execution over the same
                input set as fp_inputs.
            num_iterations: Number of iterations to optimize for during AdaScale BKD
            beta_gamma_lr: Learning rate for beta and gamma parameters
            scales_lr: Learning rate for scale parameters

        """

        compute_param_encodings(block)
        device = get_device(block)
        dtype = next(block.parameters()).dtype
        block.requires_grad_(False)
        cls._replace_with_adascale_weight_quantizers(block)
        block.to(device=device, dtype=dtype)

        def run_forward(args, kwargs):
            args = change_tensor_and_cache_device_placement(args, device)
            kwargs = change_tensor_and_cache_device_placement(kwargs, device)
            return block(*args, **kwargs)

        if not qt_inputs:
            qt_inputs = fp_inputs

        # only set adascale params to train mode
        all_beta_gamma_parameters, all_scale_parameters = (
            cls._get_adascale_trainable_params(block)
        )
        trainable_params = [
            {"params": all_beta_gamma_parameters, "lr": beta_gamma_lr},
            {"params": all_scale_parameters, "lr": scales_lr},
        ]
        optimizer = torch.optim.Adam(trainable_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations, eta_min=0.0
        )
        cls._set_requires_grad(all_beta_gamma_parameters + all_scale_parameters, True)

        fp_out = []  # save fp batchwise block outputs to use across epochs
        with remove_all_quantizers(block):
            for args, kwargs in fp_inputs:
                fp_block_results = run_forward(args, kwargs)
                fp_block_results = change_tensor_and_cache_device_placement(
                    fp_block_results, "cpu"
                )
                fp_out.append(fp_block_results)
                del args, kwargs, fp_block_results

        pbar = tqdm(
            total=num_iterations,
            leave=False,
            position=1,
            desc="Iterations completed",
        )
        curr_iteration = 0
        with remove_activation_quantizers(block):
            while curr_iteration < num_iterations:
                for batch_idx, (args, kwargs) in enumerate(qt_inputs):
                    pbar.update(1)
                    curr_iteration += 1
                    if curr_iteration > num_iterations:
                        pbar.close()
                        break
                    with torch.set_grad_enabled(True):
                        quant_out = run_forward(args, kwargs)

                        # TODO: Fix this, may not be possible to cat outputs.
                        #       Compute _LOSS_FN for each output and sum
                        if isinstance(quant_out, tuple):
                            quant_out = torch.cat(quant_out)

                        del args, kwargs

                        batch_fp_out = change_tensor_and_cache_device_placement(
                            deepcopy(fp_out[batch_idx]), device
                        )
                        if isinstance(batch_fp_out, tuple):
                            batch_fp_out = torch.cat(batch_fp_out)

                        loss = _LOSS_FN(quant_out, batch_fp_out)

                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        del quant_out, batch_fp_out, loss

    @staticmethod
    def _screen_for_target_type(model: torch.nn.Module) -> Type:
        """
        Helper to get the model type to optimize
        This is needed because the target module might not be at the top level in which case we go deeper and fetch it
        """
        for module in model.modules():
            for target in adascale_model_config_dict:
                if isinstance(module, target):
                    return target
        # No targets found in provided model
        return NoneType

    @staticmethod
    def _get_blocks(qsim: QuantizationSimModel):
        """helper to get all the blocks in the model represented by adascale_model_config_dict"""

        target_type = AdaScale._screen_for_target_type(qsim.model)
        block_type = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        ).block_type
        target_modules = []
        if block_type is not None:
            target_modules = [
                m for m in qsim.model.modules() if isinstance(m, block_type)
            ]
        return target_modules

    @staticmethod
    def _model_specific_blocks_to_disable_caching(qsim: QuantizationSimModel) -> int:
        """helper function to get the number of initial blocks for which caching should be disabled"""
        target_type = AdaScale._screen_for_target_type(qsim.model)
        num_blocks_to_disable_caching = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        ).enable_caching_after_block
        return num_blocks_to_disable_caching

    @staticmethod
    def _model_specific_lr(qsim: QuantizationSimModel) -> tuple[float, float]:
        """
        Given the sim object, query the model type and return the custom lr to be used
        """
        target_type = AdaScale._screen_for_target_type(qsim.model)
        model_config = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        )
        return model_config.beta_gamma_lr, model_config.scales_lr

    @classmethod
    def _replace_with_adascale_weight_quantizers(cls, block: torch.nn.Module):
        """Replace all the weight quantizers in supported modules with adascale quantizers"""
        for layer in block.modules():
            if isinstance(layer, tuple(supported_modules)):
                if not isinstance(layer.param_quantizers["weight"], QuantizeDequantize):
                    continue
                layer.param_quantizers["weight"] = cls._get_adascale_qdq_mapping()[
                    type(layer)
                ](layer.param_quantizers["weight"], layer.weight.shape)

    @classmethod
    @torch.no_grad()
    def fold_adascale_quantizers(cls, module: torch.nn.Module):
        for layer in module.modules():
            if isinstance(layer, tuple(supported_modules)) and isinstance(
                layer.param_quantizers["weight"], AdaScaleQuantizeDequantize
            ):
                layer.weight.copy_(
                    layer.param_quantizers["weight"].get_folded_weight(layer.weight)
                )
                layer.param_quantizers["weight"] = layer.param_quantizers[
                    "weight"
                ].get_qdq()
                layer.param_quantizers["weight"].allow_overwrite(False)
                layer.requires_grad_(False)

    @staticmethod
    def _get_adascale_trainable_params(
        non_leaf_module: torch.nn.Module,
    ) -> Tuple[List, List]:
        """Get all the adascale scale params present in the non-leaf module"""
        all_scale_parameters = []
        all_beta_gamma_parameters = []
        for module in non_leaf_module.modules():
            if isinstance(module, tuple(supported_modules)) and isinstance(
                module.param_quantizers["weight"], AdaScaleQuantizeDequantize
            ):
                beta_gamma_params, scale_parameters = module.param_quantizers[
                    "weight"
                ].get_adascale_trainable_parameters()
                all_beta_gamma_parameters.extend(beta_gamma_params)
                all_scale_parameters.extend(scale_parameters)
        return all_beta_gamma_parameters, all_scale_parameters

    @staticmethod
    def _set_requires_grad(adascale_params: list, val: bool):
        """Helper to update requires_grad to the input `val` for all the params in `adascale_params`"""
        for p in adascale_params:
            p.requires_grad = val

    @staticmethod
    def _get_adascale_qdq_mapping() -> dict:
        return {
            QuantizedLinear: AdaScaleLinearQuantizeDequantize,
            QuantizedConv2d: AdaScaleConv2dQuantizeDequantize,
        }


class _SamplingIterable:
    """
    Args
        iterable_1: First iterable to sample from
        iterable_2: Second iterable to sample from
        ratio: Ratio of values from tensors in iterable_1 to values from iterable_2 to use in the output
        device: Device to place the yielded values on
    """

    def __init__(self, iterable_1, iterable_2, ratio=0.5, device="cpu"):
        self.iterable_1 = iterable_1
        self.iterable_2 = iterable_2
        self.ratio = ratio
        self.device = device

    def __iter__(self):
        for (args_1, kwargs_1), (args_2, _) in zip(
            self.iterable_1, self.iterable_2, strict=True
        ):
            combined_args = tuple(
                self._combine_tensors(arg_1, arg_2)
                for arg_1, arg_2 in zip(
                    change_tensor_and_cache_device_placement(args_1, self.device),
                    change_tensor_and_cache_device_placement(args_2, self.device),
                    strict=True,
                )
            )

            yield (
                combined_args,
                change_tensor_and_cache_device_placement(kwargs_1, self.device),
            )

    def _combine_tensors(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor):
        return (
            torch.where(
                torch.rand_like(tensor_1, dtype=tensor_1.dtype, device=self.device)
                < self.ratio,
                tensor_1,
                tensor_2,
            )
            if isinstance(tensor_1, torch.Tensor)
            else tensor_1
        )


apply_adascale = AdaScale.apply_adascale
adascale_block = AdaScale.adascale_block
fold_adascale_quantizers = AdaScale.fold_adascale_quantizers
