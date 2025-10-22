# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-docstring

import torch
from typing import Type
from types import NoneType
from dataclasses import dataclass

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
except ImportError:
    Qwen3Model = Qwen3DecoderLayer = None

from aimet_torch.experimental.spinquant.hadamard_utils import get_hadamard_matrix
from aimet_torch.experimental.transforms.transformed_layers import TransformationMixin
from aimet_torch.experimental.transforms.transform_ops import MatrixTransformOp
from aimet_torch.experimental.transforms.transform_config import (
    BlockInterface,
    LlamaBlockInterface,
    Qwen2BlockInterface,
    Qwen3BlockInterface,
)


@dataclass
class SpinQuantConfig:
    block_type: Type = None  # block types to use in a given model
    block_interface: Type = None  # interface class describing block layout


class SpinQuant:
    model_config_dict = {
        LlamaModel: SpinQuantConfig(
            block_type=LlamaDecoderLayer, block_interface=LlamaBlockInterface
        ),
        Qwen2Model: SpinQuantConfig(
            block_type=Qwen2DecoderLayer, block_interface=Qwen2BlockInterface
        ),
    }

    if Qwen3Model is not None and Qwen3DecoderLayer is not None:
        model_config_dict.update(
            {
                Qwen3Model: SpinQuantConfig(
                    block_type=Qwen3DecoderLayer,
                    block_interface=Qwen3BlockInterface,
                )
            }
        )

    @staticmethod
    def apply_spinquant(model: torch.nn.Module):
        if model.model.embed_tokens.weight is model.lm_head.weight:
            raise RuntimeError(
                "FPTQuant requires embed_tokens and lm_head weights to be untied. Ensure that "
                "model.config.tie_word_embeddings or a similar relevant setting is set to False for the model."
            )

        # Fuse RMS norm layers into linears
        SpinQuant._fuse_model_norm_layers(model)

        # Convert all layers to transformed layers
        SpinQuant._convert_modules_to_transformed_modules(model)

        # compute R1 transform and inverse
        hidden_size = model.model.embed_tokens.weight.shape[-1]
        matrix = get_hadamard_matrix(hidden_size) / torch.sqrt(
            torch.tensor(hidden_size)
        )
        matrix = matrix.to(device=model.device, dtype=torch.float32)
        r1_transform = MatrixTransformOp(matrix=matrix)
        # note that we could obtain this module by calling r1_transform.get_inverted_op(), but that would determine
        # the matrix by doing linalg.inv, so this is an optimization given that is a hadamard matrix
        r1_transform_inverse = MatrixTransformOp(matrix=matrix.T)

        # Apply R1 transform
        model.model.embed_tokens.add_right_hand_transform(r1_transform)
        model.lm_head.add_left_hand_transform(r1_transform_inverse)
        for block_interface in SpinQuant._get_blocks(model):
            block_interface.q_proj.add_left_hand_transform(r1_transform_inverse)
            block_interface.k_proj.add_left_hand_transform(r1_transform_inverse)
            block_interface.v_proj.add_left_hand_transform(r1_transform_inverse)
            block_interface.o_proj.add_right_hand_transform(r1_transform)
            block_interface.gate_proj.add_left_hand_transform(r1_transform_inverse)
            block_interface.up_proj.add_left_hand_transform(r1_transform_inverse)
            block_interface.down_proj.add_right_hand_transform(r1_transform)

        # Convert all layers back to their original versions
        SpinQuant._merge_transforms_and_revert_to_original_modules(model)

    @staticmethod
    def _fuse_model_norm_layers(model: torch.nn.Module):
        SpinQuant._fuse_norm_layer_into_linears(model.model.norm, [model.lm_head])
        for block_interface in SpinQuant._get_blocks(model):
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.input_norm,
                [
                    block_interface.q_proj,
                    block_interface.k_proj,
                    block_interface.v_proj,
                ],
            )
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.post_attention_norm,
                [block_interface.up_proj, block_interface.gate_proj],
            )

    @staticmethod
    def _convert_modules_to_transformed_modules(model: torch.nn.Module):
        model.model.embed_tokens = TransformationMixin.from_module(
            model.model.embed_tokens
        )
        model.lm_head = TransformationMixin.from_module(model.lm_head)

        for block_interface in SpinQuant._get_blocks(model):
            block_interface.q_proj = TransformationMixin.from_module(
                block_interface.q_proj
            )
            block_interface.k_proj = TransformationMixin.from_module(
                block_interface.k_proj
            )
            block_interface.v_proj = TransformationMixin.from_module(
                block_interface.v_proj
            )
            block_interface.o_proj = TransformationMixin.from_module(
                block_interface.o_proj
            )
            block_interface.up_proj = TransformationMixin.from_module(
                block_interface.up_proj
            )
            block_interface.down_proj = TransformationMixin.from_module(
                block_interface.down_proj
            )
            block_interface.gate_proj = TransformationMixin.from_module(
                block_interface.gate_proj
            )

    @staticmethod
    def _merge_transforms_and_revert_to_original_modules(model: torch.nn.Module):
        model.model.embed_tokens = (
            SpinQuant._merge_transforms_and_recover_original_layer(
                model.model.embed_tokens
            )
        )
        model.lm_head = SpinQuant._merge_transforms_and_recover_original_layer(
            model.lm_head
        )

        for block_interface in SpinQuant._get_blocks(model):
            block_interface.q_proj = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    block_interface.q_proj
                )
            )
            block_interface.k_proj = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    block_interface.k_proj
                )
            )
            block_interface.v_proj = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    block_interface.v_proj
                )
            )
            block_interface.o_proj = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    block_interface.o_proj
                )
            )
            block_interface.up_proj = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    block_interface.up_proj
                )
            )
            block_interface.down_proj = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    block_interface.down_proj
                )
            )
            block_interface.gate_proj = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    block_interface.gate_proj
                )
            )

    @staticmethod
    def _merge_transforms_and_recover_original_layer(layer: torch.nn.Module):
        if not isinstance(layer, TransformationMixin):
            return layer  # Do nothing if it is not a transformed layer

        layer.merge()
        if len(layer.right_hand_transforms) == len(layer.left_hand_transforms) == 0:
            return TransformationMixin.get_original_module(layer)
        return layer

    @staticmethod
    def _screen_for_target_type(model: torch.nn.Module) -> Type:
        for module in model.modules():
            for target in SpinQuant.model_config_dict:
                if isinstance(module, target):
                    return target
        # No targets found in provided model
        return NoneType

    @staticmethod
    def _get_blocks(model: torch.nn.Module) -> list[BlockInterface]:
        target_type = SpinQuant._screen_for_target_type(model)
        config = SpinQuant.model_config_dict.get(target_type, SpinQuantConfig())
        target_modules = []
        if config.block_type is not None:
            target_modules = [
                config.block_interface(m)
                for m in model.modules()
                if isinstance(m, config.block_type)
            ]
        return target_modules

    @staticmethod
    def _fuse_norm_layer_into_linears(
        norm: torch.nn.Module, linears: list[torch.nn.Linear]
    ):
        """Helper function to merge RMS Norm weights into linear layer"""
        for linear in linears:
            W = linear.weight.data
            dtype = linear.weight.dtype
            linear.weight.data = (W.double() * norm.weight.data.double()).to(
                dtype=dtype
            )
            if hasattr(norm, "bias") and linear.bias is not None:
                linear.bias.data = (
                    linear.bias.data.double() + (W.double() @ norm.bias.data.double())
                ).to(dtype=dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)


apply_spinquant = SpinQuant.apply_spinquant
