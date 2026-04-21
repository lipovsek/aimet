# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level SpinQuant API for ONNX QuantizationSimModel."""

from typing import Optional, List
import numpy as np
import torch

from aimet_onnx.common.hadamard import get_hadamard_matrix
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import Op
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.experimental.spinquant.apply_rotation import (
    apply_r1_rotation,
    apply_r1_rotation_merger,
    _infer_hidden_size,
    _validate_backbone_weights,
    _validate_merger_linear2,
)
from aimet_onnx.experimental.spinquant.block_identifier import (
    DecoderModelRoleMap,
    get_decoder_block_boundaries,
    get_decoder_role_map,
    find_merger_linear2,
)
from aimet_onnx.experimental.spinquant.fuse_norm import (
    ActiveNorm,
    fuse_norm_layers_into_linears,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def apply_spinquant(
    backbone_sim: QuantizationSimModel,
    visual_sim: Optional[QuantizationSimModel] = None,
    embedding: Optional[torch.Tensor] = None,
) -> None:
    """Apply SpinQuant rotation transforms to an ONNX transformer model.

    SpinQuant applies orthogonal Hadamard rotations to model weights to reduce
    quantization error. This function modifies the QuantizationSimModel(s) in-place by:

    1. Fusing RMS normalization layers into subsequent linear layers
    2. Applying R1 Hadamard rotations to embeddings, attention, and MLP layers
    3. For VLMs: applying R_L to PatchMerger linear_fc2 (non-negotiable)

    Must be called BEFORE sim.compute_encodings(). The rotation modifies float
    weight initializers; compute_encodings must run afterward to calibrate
    quantizer scales on the rotated weights.

    Supported architectures:
        - LLaMA, Qwen2, Qwen3, Phi3 (backbone only)
        - Qwen2.5-VL, Qwen3-VL (backbone + visual)

    :param backbone_sim: QuantizationSimModel wrapping backbone.onnx.
    :param visual_sim: Optional QuantizationSimModel wrapping visual.onnx (VLM only).
    :param embedding: Optional ``torch.Tensor`` of shape ``[vocab, hidden]`` loaded
        from ``embedding.pth`` (VLM only). Rotated in-place with R_L.
    :raises ValueError: If block detection or role classification fails, if any expected
        weight is missing or has the wrong shape.

    Example (LLM)::

        sim = QuantizationSimModel(model)
        apply_spinquant(sim)
        sim.compute_encodings(calibration_data)

    Example (VLM)::

        backbone_sim = QuantizationSimModel(backbone_model)
        visual_sim = QuantizationSimModel(visual_model)
        embedding = torch.load("embedding.pth")   # torch.Tensor [vocab, hidden]
        apply_spinquant(backbone_sim, visual_sim=visual_sim, embedding=embedding)
        torch.save(embedding, "embedding.pth")    # overwrite with rotated weights
        backbone_sim.compute_encodings(backbone_calibration_data)
        visual_sim.compute_encodings(visual_calibration_data)
    """
    bb_model = backbone_sim.model.model
    bb_cg = backbone_sim.connected_graph
    bb_boundaries, bb_active_norms = get_decoder_block_boundaries(bb_model, bb_cg)
    bb_role_map = get_decoder_role_map(bb_cg, bb_boundaries, bb_active_norms)
    bb_hidden_size = _infer_hidden_size(bb_model, bb_role_map)

    if visual_sim is not None:
        v_model = visual_sim.model.model
        v_cg = visual_sim.connected_graph
        v_merger_linear2 = find_merger_linear2(v_cg)

    if embedding is not None and bb_role_map.embed_tokens:
        raise ValueError(
            "embedding was provided but backbone contains embed_tokens op(s). "
            "Pass embedding only for VLM backbones exported with use_inputs_embeds=True "
            "(i.e. backbone has no Gather op for token embeddings)."
        )
    if embedding is None and not bb_role_map.embed_tokens:
        raise ValueError(
            "Backbone has no embed_tokens op but no external embedding was provided. "
            "Pass embedding=torch.load('embedding.pth') for VLM backbones exported with "
            "use_inputs_embeds=True so the embedding weight is rotated alongside the backbone."
        )

    # Validate upfront
    _validate_backbone_weights(bb_model, bb_role_map, bb_hidden_size)
    if visual_sim is not None:
        _validate_merger_linear2(v_model, v_merger_linear2, bb_hidden_size)

    # Apply rotation
    _apply_spinquant_backbone(
        backbone_sim, bb_role_map, bb_active_norms, bb_hidden_size, embedding
    )
    if visual_sim is not None:
        _apply_spinquant_visual(visual_sim, v_merger_linear2, bb_hidden_size)


def _apply_spinquant_backbone(
    sim: QuantizationSimModel,
    role_map: DecoderModelRoleMap,
    active_norms: List[ActiveNorm],
    backbone_hidden_size: int,
    embedding: Optional[torch.Tensor] = None,
) -> None:
    """Fuse norms, apply R1 rotation, and optionally rotate external embedding in-place.

    :param sim: QuantizationSimModel wrapping backbone.onnx.
    :param role_map: Decoder role map.
    :param active_norms: Norm initializer names to fuse.
    :param backbone_hidden_size: Hidden dimension of the language model residual stream.
    :param embedding: Optional raw embedding weight tensor.
    """
    model = sim.model.model

    fuse_norm_layers_into_linears(model, active_norms)
    apply_r1_rotation(model, role_map, backbone_hidden_size)

    if embedding is not None:
        R_L = (
            get_hadamard_matrix(backbone_hidden_size) / np.sqrt(backbone_hidden_size)
        ).astype(np.float64)
        original_device = embedding.device
        W = embedding.detach().cpu().numpy()
        W_rot = (W @ R_L).astype(W.dtype)
        embedding.data.copy_(torch.from_numpy(W_rot).to(original_device))
        _logger.info(
            "Backbone: Rotated external embedding weight with R_L (shape %s).", W.shape
        )

    sim._rebuild_session()  # pylint: disable=protected-access


def _apply_spinquant_visual(
    sim: QuantizationSimModel,
    merger_linear2: List[Op],
    backbone_hidden_size: int,
) -> None:
    """Apply R_L rotation to PatchMerger linear_fc2 in the ViT visual encoder.

    :param sim: QuantizationSimModel wrapping visual.onnx.
    :param merger_linear2: List of merger_linear2 ops from find_merger_linear2.
    :param backbone_hidden_size: Hidden dimension of the language model residual stream.
    """
    model = sim.model.model

    apply_r1_rotation_merger(model, merger_linear2, backbone_hidden_size)

    sim._rebuild_session()  # pylint: disable=protected-access
