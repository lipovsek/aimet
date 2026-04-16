# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import torch
from typing import Type, List
from dataclasses import dataclass

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3Model, Phi3DecoderLayer

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
except ImportError:
    Qwen3Model = Qwen3DecoderLayer = None

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLTextModel,
        Qwen2_5_VLDecoderLayer,
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VLPatchMerger,
    )
except ImportError:
    Qwen2_5_VLTextModel = Qwen2_5_VLDecoderLayer = (
        Qwen2_5_VisionTransformerPretrainedModel
    ) = Qwen2_5_VLVisionBlock = Qwen2_5_VLPatchMerger = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextModel,
        Qwen3VLTextDecoderLayer,
        Qwen3VLVisionPatchMerger,
    )
except ImportError:
    Qwen3VLTextModel = Qwen3VLTextDecoderLayer = Qwen3VLVisionPatchMerger = None


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
    Qwen3VLBackboneBlockInterface,
    MergerInterface,
    Qwen25VLMergerInterface,
    Qwen3VLMergerInterface,
)


@dataclass
class SpinQuantConfig:
    block_type: Type = None  # block types to use in a given model
    block_interface: Type = None  # interface class describing block layout


@dataclass
class VLMSpinQuantConfig(SpinQuantConfig):
    merger_type: Type = None  # merger type to use for VLMs, if applicable
    merger_interface: Type[MergerInterface] = (
        None  # interface class describing merger layout for VLMs
    )


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
    }
    if Qwen2_5_VLTextModel is not None and Qwen2_5_VLDecoderLayer is not None:
        model_config_dict.update(
            {
                Qwen2_5_VLTextModel: VLMSpinQuantConfig(
                    block_type=Qwen2_5_VLDecoderLayer,
                    block_interface=Qwen2dot5VLBackboneBlockInterface,
                    merger_type=Qwen2_5_VLPatchMerger,
                    merger_interface=Qwen25VLMergerInterface,
                )
            }
        )
    if (
        Qwen2_5_VisionTransformerPretrainedModel is not None
        and Qwen2_5_VLVisionBlock is not None
    ):
        model_config_dict.update(
            {
                Qwen2_5_VisionTransformerPretrainedModel: SpinQuantConfig(
                    block_type=Qwen2_5_VLVisionBlock,
                    block_interface=Qwen2dot5VLViTBlockInterface,
                )
            }
        )
    if Qwen3Model is not None and Qwen3DecoderLayer is not None:
        model_config_dict.update(
            {
                Qwen3Model: SpinQuantConfig(
                    block_type=Qwen3DecoderLayer,
                    block_interface=Qwen3BlockInterface,
                )
            }
        )
    if Qwen3VLTextModel is not None and Qwen3VLTextDecoderLayer is not None:
        model_config_dict.update(
            {
                Qwen3VLTextModel: VLMSpinQuantConfig(
                    block_type=Qwen3VLTextDecoderLayer,
                    block_interface=Qwen3VLBackboneBlockInterface,
                    merger_type=Qwen3VLVisionPatchMerger,
                    merger_interface=Qwen3VLMergerInterface,
                )
            }
        )

    _enable_r1 = True
    _enable_r2 = False

    @staticmethod
    def apply_spinquant(model: torch.nn.Module):
        """
        Apply SpinQuant rotation transforms to a transformer-based language model.

        SpinQuant applies orthogonal Hadamard rotations to model weights to reduce
        quantization error. This method modifies the model in-place by:

        1. Fusing RMS normalization layers into subsequent linear layers
        2. Applying R1 Hadamard rotations to embeddings, attention, and MLP layers
        3. Merging all transforms into the weight matrices

        Supported architectures:
            - LLaMA
            - Qwen2, Qwen3
            - Phi3
            - Qwen2.5-VL (Vision-Language Model)

        :param model: A HuggingFace transformer model (e.g., LlamaForCausalLM,
            Qwen2ForCausalLM). The model must have untied embed_tokens and lm_head
            weights.
        :raises RuntimeError: If embed_tokens and lm_head weights are tied.

        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> from aimet_torch.experimental.spinquant import apply_spinquant
            >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
            >>> # Untie embedding and lm_head weights if they are tied
            >>> old_weight = model.lm_head.weight
            >>> model.lm_head.weight = torch.nn.Parameter(
            ...     old_weight.data.clone().detach().to(old_weight.device),
            ...     requires_grad=old_weight.requires_grad,
            ... )
            >>> apply_spinquant(model)
        """
        language_backbone = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )

        SpinQuant._apply_spinquant_to_decoder_stack(language_backbone, model.lm_head)

        if hasattr(model.model, "visual"):
            language_backbone_hidden_size = language_backbone.embed_tokens.weight.shape[
                -1
            ]
            vlm_config = SpinQuant._find_vlm_config(model)
            merger_modules = (
                [m for m in model.modules() if isinstance(m, vlm_config.merger_type)]
                if vlm_config
                else None
            )
            merger_interface_cls = vlm_config.merger_interface if vlm_config else None

            SpinQuant._apply_spinquant_to_visual_encoder(
                visual_encoder=model.model.visual,
                language_backbone_hidden_size=language_backbone_hidden_size,
                merger_modules=merger_modules,
                merger_interface=merger_interface_cls,
            )

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
            if hasattr(model.model.visual, "pos_embed"):
                model.model.visual.pos_embed = TransformationMixin.from_module(
                    model.model.visual.pos_embed
                )

            for merger in SpinQuant._get_mergers(model):
                merger.linear1 = TransformationMixin.from_module(merger.linear1)
                merger.linear2 = TransformationMixin.from_module(merger.linear2)

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
            for merger in SpinQuant._get_mergers(model):
                merger.linear1 = SpinQuant._merge_transforms_and_recover_original_layer(
                    merger.linear1
                )
                merger.linear2 = SpinQuant._merge_transforms_and_recover_original_layer(
                    merger.linear2
                )
            if hasattr(model.model.visual, "pos_embed"):
                model.model.visual.pos_embed = (
                    SpinQuant._merge_transforms_and_recover_original_layer(
                        model.model.visual.pos_embed
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
            B = linear.bias.data if linear.bias is not None else None
            dtype = linear.weight.dtype

            if norm.weight.data.shape[0] != W.shape[1]:
                norm_weight = norm.weight.data.repeat(
                    W.shape[1] // norm.weight.data.shape[0]
                )
                norm_bias = (
                    norm.bias.data.repeat(B.shape[0] // norm.bias.data.shape[0])
                    if hasattr(norm, "bias")
                    else None
                )
            else:
                norm_weight = norm.weight.data
                norm_bias = norm.bias.data if hasattr(norm, "bias") else None

            linear.weight.data = (W.double() * norm_weight.double()).to(dtype=dtype)
            if hasattr(norm, "bias") and linear.bias is not None:
                linear.bias.data = (B.double() + (W.double() @ norm_bias.double())).to(
                    dtype=dtype
                )
        norm.weight.data = torch.ones_like(norm.weight.data)

    @staticmethod
    def _find_vlm_config(model: torch.nn.Module) -> VLMSpinQuantConfig | None:
        target_types = SpinQuant._screen_for_target_type(model)
        for target_type in target_types:
            config = SpinQuant.model_config_dict.get(target_type)
            if config is not None and isinstance(config, VLMSpinQuantConfig):
                return config

        # Fallback: look for a VLMSpinQuantConfig whose merger_type is present
        # in the model (e.g. when called with just the visual encoder).
        module_types = {type(m) for m in model.modules()}
        for config in SpinQuant.model_config_dict.values():
            if (
                isinstance(config, VLMSpinQuantConfig)
                and config.merger_type in module_types
            ):
                return config

        return None

    @staticmethod
    def _get_mergers(model: torch.nn.Module) -> list[MergerInterface]:
        vlm_config = SpinQuant._find_vlm_config(model)
        return [
            vlm_config.merger_interface(m)
            for m in model.modules()
            if isinstance(m, vlm_config.merger_type)
        ]

    @staticmethod
    def _wrap_mergers(
        merger_modules: list[torch.nn.Module],
        merger_interface: Type[MergerInterface],
    ) -> list[MergerInterface]:
        return [merger_interface(m) for m in merger_modules]

    @staticmethod
    def _convert_block_layers_to_transformed(
        blocks: list[BlockInterface],
    ) -> None:
        for block_interface in blocks:
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
    def _merge_and_revert_block_layers(
        blocks: list[BlockInterface],
    ) -> None:
        for block_interface in blocks:
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
    def _apply_spinquant_to_decoder_stack(
        decoder_model: torch.nn.Module,
        lm_head: torch.nn.Module,
    ) -> None:
        """
        Apply SpinQuant rotation transforms to a decoder-only language backbone.

        This processes the language backbone in isolation, applying norm fusion and
        R1 Hadamard rotations to embeddings, attention, and MLP layers.

        :param decoder_model: Language backbone with .embed_tokens, .norm, and decoder
            blocks (e.g., model.model for LlamaForCausalLM).
        :param lm_head: The lm_head linear layer (e.g., model.lm_head).
        :raises RuntimeError: If embed_tokens and lm_head weights are tied.
        """
        if decoder_model.embed_tokens.weight is lm_head.weight:
            raise RuntimeError(
                "SpinQuant requires embed_tokens and lm_head weights to be untied. Ensure that "
                "model.config.tie_word_embeddings or a similar relevant setting is set to False for the model."
            )

        # Fuse RMS norm layers into linears
        SpinQuant._fuse_norm_layer_into_linears(decoder_model.norm, [lm_head])
        blocks = SpinQuant._get_blocks(decoder_model)
        for block_interface in blocks:
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.input_norm,
                list(block_interface.qkv_layers()),
            )
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.post_attention_norm,
                list(block_interface.gate_up_layers()),
            )

        # Convert to transformed modules
        decoder_model.embed_tokens = TransformationMixin.from_module(
            decoder_model.embed_tokens
        )
        lm_head_transformed = TransformationMixin.from_module(lm_head)
        # Update lm_head in-place by copying transformed state
        # We need to work with the transformed wrapper directly
        SpinQuant._convert_block_layers_to_transformed(blocks)

        if SpinQuant._enable_r1:
            hidden_size = decoder_model.embed_tokens.weight.shape[-1]
            device = lm_head.weight.device
            SpinQuant._apply_r1_to_decoder_stack(
                hidden_size=hidden_size,
                embedding_layer=decoder_model.embed_tokens,
                lm_head=lm_head_transformed,
                blocks=blocks,
                device=device,
            )

        if SpinQuant._enable_r2:
            raise NotImplementedError("R2 transform is not yet implemented.")

        # Merge transforms and revert
        decoder_model.embed_tokens = (
            SpinQuant._merge_transforms_and_recover_original_layer(
                decoder_model.embed_tokens
            )
        )
        merged_lm_head = SpinQuant._merge_transforms_and_recover_original_layer(
            lm_head_transformed
        )
        # Copy merged weight/bias back into the original lm_head module
        lm_head.weight = merged_lm_head.weight
        if hasattr(merged_lm_head, "bias") and merged_lm_head.bias is not None:
            lm_head.bias = merged_lm_head.bias
        SpinQuant._merge_and_revert_block_layers(blocks)

    @staticmethod
    def _apply_spinquant_to_visual_encoder(
        visual_encoder: torch.nn.Module,
        language_backbone_hidden_size: int,
        merger_modules: list[torch.nn.Module] | None = None,
        merger_interface: Type[MergerInterface] | None = None,
    ) -> None:
        """
        Apply SpinQuant rotation transforms to a visual encoder (ViT) and its merger modules.

        :param visual_encoder: Visual transformer module with .patch_embed.proj and ViT blocks.
        :param language_backbone_hidden_size: Hidden size of the language backbone, needed for
            the post-merger Hadamard rotation on merger output layers.
        :param merger_modules: Raw merger nn.Module instances. If None, mergers are
            auto-discovered by scanning visual_encoder.
        :param merger_interface: MergerInterface subclass to wrap merger_modules. Required if
            merger_modules is provided.
        """
        # Resolve mergers
        if merger_modules is not None:
            if merger_interface is None:
                raise ValueError(
                    "merger_interface must be provided when merger_modules is specified."
                )
            mergers = SpinQuant._wrap_mergers(merger_modules, merger_interface)
        else:
            mergers = []
            vlm_config = SpinQuant._find_vlm_config(visual_encoder)
            if vlm_config is not None:
                mergers = [
                    vlm_config.merger_interface(m)
                    for m in visual_encoder.modules()
                    if isinstance(m, vlm_config.merger_type)
                ]

        # Fuse merger norms
        for merger in mergers:
            SpinQuant._fuse_norm_layer_into_linears(merger.norm, [merger.linear1])

        # Fuse ViT block norms
        vit_blocks = SpinQuant._get_blocks(visual_encoder)
        for block_interface in vit_blocks:
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.input_norm,
                list(block_interface.qkv_layers()),
            )
            SpinQuant._fuse_norm_layer_into_linears(
                block_interface.post_attention_norm,
                list(block_interface.gate_up_layers()),
            )

        # Convert to transformed modules
        if hasattr(visual_encoder, "pos_embed"):
            visual_encoder.pos_embed = TransformationMixin.from_module(
                visual_encoder.pos_embed
            )
        for merger in mergers:
            merger.linear1 = TransformationMixin.from_module(merger.linear1)
            merger.linear2 = TransformationMixin.from_module(merger.linear2)
        SpinQuant._convert_block_layers_to_transformed(vit_blocks)

        if SpinQuant._enable_r1:
            device = next(visual_encoder.parameters()).device

            # Apply R1 to ViT blocks if registered
            if len(vit_blocks) > 0:
                vit_hidden_size = visual_encoder.patch_embed.proj.weight.shape[0]
                SpinQuant._apply_r1_to_decoder_stack(
                    hidden_size=vit_hidden_size,
                    embedding_layer=None,
                    lm_head=None,
                    blocks=vit_blocks,
                    device=device,
                )

                # Rotate patch_embed.proj weight directly
                patch_embed_shape = visual_encoder.patch_embed.proj.weight.shape
                patch_embed_hadamard_rotation = get_hadamard_matrix(
                    vit_hidden_size
                ) / torch.sqrt(torch.tensor(vit_hidden_size))
                new_patch_embed_weight = (
                    visual_encoder.patch_embed.proj.weight.data.clone()
                    .detach()
                    .to(torch.float64)
                    .reshape([vit_hidden_size, -1])
                )
                new_patch_embed_weight = (
                    patch_embed_hadamard_rotation.T.to(
                        device=device, dtype=torch.float64
                    )
                    @ new_patch_embed_weight
                ).to(torch.float32)
                visual_encoder.patch_embed.proj.weight = torch.nn.Parameter(
                    new_patch_embed_weight.reshape(patch_embed_shape)
                )

                # Rotate merger.linear1 weights
                for merger in mergers:
                    mlp0_shape = merger.linear1.weight.data.shape
                    new_mlp0_weight = (
                        merger.linear1.weight.data.clone()
                        .detach()
                        .to(torch.float64)
                        .reshape(-1, vit_hidden_size)
                    )
                    new_mlp0_weight = (
                        new_mlp0_weight
                        @ patch_embed_hadamard_rotation.T.to(
                            device=device, dtype=torch.float64
                        )
                    ).to(dtype=torch.float32)
                    merger.linear1.weight = torch.nn.Parameter(
                        new_mlp0_weight.reshape(mlp0_shape)
                    )

            # Post-MLP Hadamard rotation on merger.linear2 (always, even without ViT blocks)
            post_mlp_hadamard_rotation = get_hadamard_matrix(
                language_backbone_hidden_size
            ) / torch.sqrt(torch.tensor(language_backbone_hidden_size))
            post_mlp_hadamard_rotation = post_mlp_hadamard_rotation.to(
                device=device, dtype=torch.float32
            )
            for merger in mergers:
                merger.linear2.add_right_hand_transform(
                    MatrixTransformOp(matrix=post_mlp_hadamard_rotation)
                )

        if SpinQuant._enable_r2:
            raise NotImplementedError("R2 transform is not yet implemented.")

        # Merge transforms and revert
        if hasattr(visual_encoder, "pos_embed"):
            visual_encoder.pos_embed = (
                SpinQuant._merge_transforms_and_recover_original_layer(
                    visual_encoder.pos_embed
                )
            )
        for merger in mergers:
            merger.linear1 = SpinQuant._merge_transforms_and_recover_original_layer(
                merger.linear1
            )
            merger.linear2 = SpinQuant._merge_transforms_and_recover_original_layer(
                merger.linear2
            )
        SpinQuant._merge_and_revert_block_layers(vit_blocks)

    @staticmethod
    def _apply_spinquant_to_embedding(
        embedding_module: torch.nn.Embedding,
        hidden_size: int,
        device: torch.device | None = None,
    ) -> torch.nn.Embedding:
        """
        Apply R1 Hadamard rotation to a standalone embedding table.

        :param embedding_module: The embedding module to rotate.
        :param hidden_size: Hidden size for the Hadamard matrix.
        :param device: Device for computation. Inferred from embedding if None.
        :return: The rotated embedding module.
        """
        if device is None:
            device = embedding_module.weight.device

        transformed = TransformationMixin.from_module(embedding_module)

        matrix = get_hadamard_matrix(hidden_size) / torch.sqrt(
            torch.tensor(hidden_size)
        )
        matrix = matrix.to(device=device, dtype=torch.float32)
        r1_transform = MatrixTransformOp(matrix=matrix)
        transformed.add_right_hand_transform(r1_transform)

        return SpinQuant._merge_transforms_and_recover_original_layer(transformed)


apply_spinquant = SpinQuant.apply_spinquant
