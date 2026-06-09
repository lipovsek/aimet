# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level SpinQuant API for ONNX QuantizationSimModel.

This module is the entry-point orchestrator. It performs model analysis once,
builds a :class:`SpinquantContext`, then runs the rotation passes selected by
the caller via boolean flags (``enable_r1`` / ``enable_r2``).
"""

from typing import List, Optional

import onnx
import torch

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph

from aimet_onnx.experimental.spinquant.model_analysis import (
    find_merger_linear2,
    get_decoder_block_boundaries,
    get_decoder_role_map,
    infer_head_dim,
    infer_hidden_size,
)
from aimet_onnx.experimental.spinquant.passes import (
    R1RotationPass,
    R2RotationPass,
    R3RotationPass,
    RotationPass,
    SpinquantContext,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def apply_spinquant(
    model: onnx.ModelProto,
    visual_model: Optional[onnx.ModelProto] = None,
    embedding: Optional[torch.Tensor] = None,
    *,
    enable_r1: bool = True,
    enable_r2: bool = False,
    enable_r3: bool = False,
) -> None:
    """Apply SpinQuant rotation transforms to an ONNX transformer model.

    SpinQuant applies orthogonal Hadamard rotations to model weights to reduce
    quantization error. This function modifies the ONNX model(s) in-place by:

    1. Analyzing the backbone (decoder block boundaries, role map, hidden size)
       and the optional visual encoder (PatchMerger output projection).
    2. Validating every selected rotation pass against the analysis.
    3. Applying every selected pass in order (R1 before R2 before R3).

    Must be called on the float ONNX model BEFORE creating a
    ``QuantizationSimModel``. The rotation modifies float weight initializers
    (R1 / R2) and may insert new nodes (R3); build the sim on the rotated graph
    and run ``compute_encodings`` afterward so quantizer scales are calibrated on
    the rotated weights.

    Supported architectures:
        - LLaMA, Qwen2, Qwen3, Phi3 (backbone only)
        - Qwen2.5-VL, Qwen3-VL (backbone + visual)

    :param model: backbone.onnx ModelProto. Mutated in-place.
    :param visual_model: Optional visual.onnx ModelProto (VLM only). Mutated in-place.
    :param embedding: Optional ``torch.Tensor`` of shape ``[vocab, hidden]`` loaded
        from ``embedding.pth`` (VLM only). Rotated in-place with R_L.
    :param enable_r1: If ``True`` (default), apply the R1 (residual-stream) rotation.
    :param enable_r2: If ``True``, apply the R2 (per-head) rotation. Defaults to ``False``.
        Not supported on architectures with fused QKV projections (e.g. Phi3).
    :param enable_r3: If ``True``, apply the R3 online Hadamard rotation on Q and K
        paths into each block's QK^T MatMul. Defaults to ``False``. Inserts new
        ``MatMul`` nodes into the ONNX graph (does not mutate existing weights).
        MHA only — not supported on fused QKV or per-head split exports. The K-side
        rotation is placed upstream of the past-key ``Concat`` so K values entering
        the KV cache are already rotated; the model's ``present_key`` output then
        carries rotated K (cache convention is self-consistent across steps).
    :raises ValueError: If no rotation is enabled, if block detection or role
        classification fails, or if any expected weight is missing / has the wrong shape.

    Example (LLM)::

        apply_spinquant(model)
        sim = QuantizationSimModel(model)
        sim.compute_encodings(calibration_data)

    Example (VLM)::

        embedding = torch.load("embedding.pth")   # torch.Tensor [vocab, hidden]
        apply_spinquant(backbone_model, visual_model=visual_model, embedding=embedding)
        torch.save(embedding, "embedding.pth")    # overwrite with rotated weights
        backbone_sim = QuantizationSimModel(backbone_model)
        visual_sim = QuantizationSimModel(visual_model)
        backbone_sim.compute_encodings(backbone_calibration_data)
        visual_sim.compute_encodings(visual_calibration_data)
    """
    rotations: List[RotationPass] = []
    if enable_r1:
        rotations.append(R1RotationPass())
    if enable_r2:
        rotations.append(R2RotationPass())
    if enable_r3:
        rotations.append(R3RotationPass())
    if not rotations:
        raise ValueError(
            "apply_spinquant requires at least one rotation enabled "
            "(set enable_r1=True and/or enable_r2=True and/or enable_r3=True)."
        )

    ctx = _build_context(model, visual_model, embedding)

    # Validate every pass before mutating anything: a bad config must not
    # leave the model half-rotated.
    for rotation in rotations:
        rotation.validate(ctx)

    for rotation in rotations:
        _logger.info("Applying %s rotation pass.", rotation.name)
        rotation.apply(ctx)


def _build_context(
    model: onnx.ModelProto,
    visual_model: Optional[onnx.ModelProto],
    embedding: Optional[torch.Tensor],
) -> SpinquantContext:
    """Run model analysis once and build the context shared across passes."""
    bb_cg = ConnectedGraph(model)
    boundaries, active_norms = get_decoder_block_boundaries(model, bb_cg)
    role_map = get_decoder_role_map(bb_cg, boundaries, active_norms)
    hidden_size = infer_hidden_size(model, role_map)

    # head_dim is only needed by R2; derivation requires a past_value graph
    # input. Tolerate missing KV-cache inputs here so non-R2 flows still work,
    # and let R2.validate raise a targeted error when it actually needs the value.
    try:
        head_dim = infer_head_dim(model)
    except ValueError:
        head_dim = None

    visual_merger_linear2 = None
    if visual_model is not None:
        visual_merger_linear2 = find_merger_linear2(ConnectedGraph(visual_model))

    _check_embedding_consistency(role_map, embedding)

    return SpinquantContext(
        backbone_model=model,
        backbone_role_map=role_map,
        backbone_active_norms=active_norms,
        backbone_hidden_size=hidden_size,
        backbone_head_dim=head_dim,
        visual_model=visual_model,
        visual_merger_linear2=visual_merger_linear2,
        embedding=embedding,
    )


def _check_embedding_consistency(role_map, embedding: Optional[torch.Tensor]) -> None:
    """Reject inconsistent (embedding, embed_tokens) combinations.

    A backbone exported with ``use_inputs_embeds=True`` has no Gather op for
    token embeddings; the caller must pass the external embedding tensor so it
    gets rotated alongside the backbone. Conversely, a backbone with embed_tokens
    must NOT receive an external embedding (it would be rotated twice).
    """
    if embedding is not None and role_map.embed_tokens:
        raise ValueError(
            "embedding was provided but backbone contains embed_tokens op(s). "
            "Pass embedding only for VLM backbones exported with use_inputs_embeds=True "
            "(i.e. backbone has no Gather op for token embeddings)."
        )
    if embedding is None and not role_map.embed_tokens:
        raise ValueError(
            "Backbone has no embed_tokens op but no external embedding was provided. "
            "Pass embedding=torch.load('embedding.pth') for VLM backbones exported with "
            "use_inputs_embeds=True so the embedding weight is rotated alongside the backbone."
        )
