# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-docstring

import torch
from typing import Type, List
from dataclasses import dataclass

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3Model, Phi3DecoderLayer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextModel,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLVisionBlock,
)

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
    Phi3BlockInterface,
    Qwen2dot5VLViTBlockInterface,
    Qwen2dot5VLBackboneBlockInterface,
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
        Phi3Model: SpinQuantConfig(
            block_type=Phi3DecoderLayer, block_interface=Phi3BlockInterface
        ),
        Qwen2_5_VLTextModel: SpinQuantConfig(
            block_type=Qwen2_5_VLDecoderLayer,
            block_interface=Qwen2dot5VLBackboneBlockInterface,
        ),
        Qwen2_5_VisionTransformerPretrainedModel: SpinQuantConfig(
            block_type=Qwen2_5_VLVisionBlock,
            block_interface=Qwen2dot5VLViTBlockInterface,
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

    _enable_r1 = True
    _enable_r2 = False

    @staticmethod
    def apply_spinquant(model: torch.nn.Module):
        language_backbone = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )
        if language_backbone.embed_tokens.weight is model.lm_head.weight:
            raise RuntimeError(
                "SpinQuant requires embed_tokens and lm_head weights to be untied. Ensure that "
                "model.config.tie_word_embeddings or a similar relevant setting is set to False for the model."
            )

        # Fuse RMS norm layers into linears
        SpinQuant._fuse_norm_layer_into_linears(language_backbone.norm, [model.lm_head])
        if hasattr(model.model, "visual"):
            SpinQuant._fuse_norm_layer_into_linears(
                model.model.visual.merger.ln_q, [model.model.visual.merger.mlp[0]]
            )

        for block_interface in SpinQuant._get_blocks(model):
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.input_norm,
                list(block_interface.qkv_layers()),
            )
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.post_attention_norm,
                list(block_interface.gate_up_layers()),
            )

        # Convert all layers to transformed layers
        SpinQuant._convert_modules_to_transformed_modules(model)

        if SpinQuant._enable_r1:
            # Apply R1 to language backbone
            language_backbone_hidden_size = language_backbone.embed_tokens.weight.shape[
                -1
            ]
            SpinQuant._apply_r1_to_decoder_stack(
                hidden_size=language_backbone_hidden_size,
                embedding_layer=language_backbone.embed_tokens,
                lm_head=model.lm_head,
                blocks=SpinQuant._get_blocks(language_backbone),
                device=model.device,
            )

            # If SpinQuant is being applied to a VLM we need to separately transform the ViT
            if hasattr(model.model, "visual"):
                # Apply R1 to ViT
                vit_hidden_size = model.model.visual.patch_embed.proj.weight.shape[0]
                SpinQuant._apply_r1_to_decoder_stack(
                    hidden_size=vit_hidden_size,
                    embedding_layer=None,
                    lm_head=None,
                    blocks=SpinQuant._get_blocks(model.model.visual),
                    device=model.device,
                )

                # todo: switch this to use TransformedConv3d once implementation is tested
                patch_embed_shape = model.model.visual.patch_embed.proj.weight.shape
                patch_embed_hadamard_rotation = get_hadamard_matrix(
                    vit_hidden_size
                ) / torch.sqrt(torch.tensor(vit_hidden_size))
                new_patch_embed_weight = (
                    model.model.visual.patch_embed.proj.weight.data.clone()
                    .detach()
                    .to(torch.float64)
                    .reshape([vit_hidden_size, -1])
                )
                new_patch_embed_weight = (
                    patch_embed_hadamard_rotation.T.to(
                        device=model.device, dtype=torch.float64
                    )
                    @ new_patch_embed_weight
                ).to(torch.float32)
                model.model.visual.patch_embed.proj.weight = torch.nn.Parameter(
                    new_patch_embed_weight.reshape(patch_embed_shape)
                )

                # ViT MLP layer is sized differently so we need to handle it separately
                mlp0_shape = model.model.visual.merger.mlp[0].weight.data.shape
                new_mlp0_weight = (
                    model.model.visual.merger.mlp[0]
                    .weight.data.clone()
                    .detach()
                    .to(torch.float64)
                    .reshape(-1, vit_hidden_size)
                )
                new_mlp0_weight = (
                    new_mlp0_weight
                    @ patch_embed_hadamard_rotation.T.to(
                        device=model.device, dtype=torch.float64
                    )
                ).to(dtype=torch.float32)
                model.model.visual.merger.mlp[0].weight = torch.nn.Parameter(
                    new_mlp0_weight.reshape(mlp0_shape)
                )

                post_mlp_hadamard_rotation = get_hadamard_matrix(
                    language_backbone_hidden_size
                ) / torch.sqrt(torch.tensor(language_backbone_hidden_size))
                post_mlp_hadamard_rotation = post_mlp_hadamard_rotation.to(
                    device=model.device, dtype=torch.float32
                )
                model.model.visual.merger.mlp[-1].add_right_hand_transform(
                    MatrixTransformOp(matrix=post_mlp_hadamard_rotation)
                )

        if SpinQuant._enable_r2:
            raise NotImplementedError("R2 transform is not yet implemented.")

        # Convert all layers back to their original versions
        SpinQuant._merge_transforms_and_revert_to_original_modules(model)

    @staticmethod
    def _apply_r1_to_decoder_stack(
        hidden_size: int,
        embedding_layer: torch.nn.Module | None,
        lm_head: torch.nn.Module | None,
        blocks: List[BlockInterface],
        device: torch.device,
    ):
        matrix = get_hadamard_matrix(hidden_size) / torch.sqrt(
            torch.tensor(hidden_size)
        )
        matrix = matrix.to(device=device, dtype=torch.float32)

        # compute R1 transform and inverse
        r1_transform = MatrixTransformOp(matrix=matrix)

        # note that we could obtain this module by calling r1_transform.get_inverted_op(), but that would determine
        # the matrix by doing linalg.inv, so this is an optimization given that is a hadamard matrix
        r1_transform_inverse = MatrixTransformOp(matrix=matrix.T)

        if embedding_layer is not None:
            embedding_layer.add_right_hand_transform(r1_transform)

        if lm_head is not None:
            lm_head.add_left_hand_transform(r1_transform_inverse)

        for block_interface in blocks:
            for layer in block_interface.qkv_layers():
                layer.add_left_hand_transform(r1_transform_inverse)
            block_interface.o_proj.add_right_hand_transform(r1_transform)

            for layer in block_interface.gate_up_layers():
                layer.add_left_hand_transform(r1_transform_inverse)
            block_interface.down_proj.add_right_hand_transform(r1_transform)

    @staticmethod
    def _convert_modules_to_transformed_modules(model: torch.nn.Module):
        language_backbone = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )
        language_backbone.embed_tokens = TransformationMixin.from_module(
            language_backbone.embed_tokens
        )
        model.lm_head = TransformationMixin.from_module(model.lm_head)

        if hasattr(model.model, "visual"):
            # todo: re-enable this once TransformedConv3d is tested
            # model.model.visual.patch_embed.proj = TransformationMixin.from_module(
            #    model.model.visual.patch_embed.proj
            # )
            model.model.visual.merger.mlp[0] = TransformationMixin.from_module(
                model.model.visual.merger.mlp[0]
            )
            model.model.visual.merger.mlp[2] = TransformationMixin.from_module(
                model.model.visual.merger.mlp[2]
            )

        for block_interface in SpinQuant._get_blocks(model):
            layer_names = (
                block_interface.attention_layer_names()
                + block_interface.mlp_layer_names()
            )
            for layer_name in layer_names:
                layer = getattr(block_interface, layer_name)
                if not isinstance(layer, TransformationMixin):
                    setattr(
                        block_interface,
                        layer_name,
                        TransformationMixin.from_module(layer),
                    )

    @staticmethod
    def _merge_transforms_and_revert_to_original_modules(model: torch.nn.Module):
        language_backbone = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )
        language_backbone.embed_tokens = (
            SpinQuant._merge_transforms_and_recover_original_layer(
                language_backbone.embed_tokens
            )
        )
        model.lm_head = SpinQuant._merge_transforms_and_recover_original_layer(
            model.lm_head
        )

        if hasattr(model.model, "visual"):
            # todo: re-enable this once TransformedConv3d is tested
            # model.model.visual.patch_embed.proj = (
            #    SpinQuant._merge_transforms_and_recover_original_layer(
            #        model.model.visual.patch_embed.proj
            #    )
            # )
            model.model.visual.merger.mlp[0] = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    model.model.visual.merger.mlp[0]
                )
            )
            model.model.visual.merger.mlp[2] = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    model.model.visual.merger.mlp[2]
                )
            )

        for block_interface in SpinQuant._get_blocks(model):
            layer_names = (
                block_interface.attention_layer_names()
                + block_interface.mlp_layer_names()
            )
            for layer_name in layer_names:
                layer = getattr(block_interface, layer_name)
                if isinstance(layer, TransformationMixin):
                    merged_layer = (
                        SpinQuant._merge_transforms_and_recover_original_layer(layer)
                    )
                    setattr(block_interface, layer_name, merged_layer)

    @staticmethod
    def _merge_transforms_and_recover_original_layer(layer: torch.nn.Module):
        if not isinstance(layer, TransformationMixin):
            return layer  # Do nothing if it is not a transformed layer

        layer.merge()
        if len(layer.right_hand_transforms) == len(layer.left_hand_transforms) == 0:
            return TransformationMixin.get_original_module(layer)
        return layer

    @staticmethod
    def _screen_for_target_type(model: torch.nn.Module) -> List[Type]:
        found_targets = []
        for module in model.modules():
            for target in SpinQuant.model_config_dict:
                if isinstance(module, target):
                    found_targets.append(target)
        return found_targets

    @staticmethod
    def _get_blocks(model: torch.nn.Module) -> list[BlockInterface]:
        target_types = SpinQuant._screen_for_target_type(model)
        target_modules = []
        for target_type in target_types:
            config = SpinQuant.model_config_dict.get(target_type, SpinQuantConfig())
            if config.block_type is not None:
                target_modules.extend(
                    list(
                        config.block_interface(m)
                        for m in model.modules()
                        if isinstance(m, config.block_type)
                    )
                )
        return target_modules

    @staticmethod
    def _fuse_norm_layer_into_linears(
        norm: torch.nn.Module, linears: list[torch.nn.Linear]
    ):
        """Helper function to merge RMS Norm weights into linear layer"""
        for linear in linears:
            W = linear.weight.data
            dtype = linear.weight.dtype

            if norm.weight.data.shape[0] != W.shape[1]:
                norm_weight = norm.weight.data.repeat(
                    W.shape[1] // norm.weight.data.shape[0]
                )
            else:
                norm_weight = norm.weight.data

            linear.weight.data = (W.double() * norm_weight.double()).to(dtype=dtype)
            if hasattr(norm, "bias") and linear.bias is not None:
                linear.bias.data = (
                    linear.bias.data.double() + (W.double() @ norm.bias.data.double())
                ).to(dtype=dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)


apply_spinquant = SpinQuant.apply_spinquant
