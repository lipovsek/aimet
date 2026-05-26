# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level SpinQuant API for ONNX QuantizationSimModel.

This module is the entry-point orchestrator. It performs model analysis once,
builds a :class:`SpinquantContext`, then runs a list of :class:`RotationPass`
objects (currently just :class:`R1RotationPass`; R2/R3 will plug in here).
"""

from typing import Optional, Sequence

import torch

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.quantsim import QuantizationSimModel

from aimet_onnx.experimental.spinquant.model_analysis import (
    find_merger_linear2,
    get_decoder_block_boundaries,
    get_decoder_role_map,
    infer_hidden_size,
)
from aimet_onnx.experimental.spinquant.passes import (
    R1RotationPass,
    RotationPass,
    SpinquantContext,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def apply_spinquant(
    backbone_sim: QuantizationSimModel,
    visual_sim: Optional[QuantizationSimModel] = None,
    embedding: Optional[torch.Tensor] = None,
    *,
    rotations: Optional[Sequence[RotationPass]] = None,
) -> None:
    """Apply SpinQuant rotation transforms to an ONNX transformer model.

    SpinQuant applies orthogonal Hadamard rotations to model weights to reduce
    quantization error. This function modifies the QuantizationSimModel(s) in-place by:

    1. Analyzing the backbone (decoder block boundaries, role map, hidden size)
       and the optional visual encoder (PatchMerger output projection).
    2. Validating every requested rotation pass against the analysis.
    3. Applying every requested pass in order.
    4. Rebuilding ORT sessions once per sim.

    Must be called BEFORE ``sim.compute_encodings()``. The rotation modifies float
    weight initializers; ``compute_encodings`` must run afterward to calibrate
    quantizer scales on the rotated weights.

    Supported architectures:
        - LLaMA, Qwen2, Qwen3, Phi3 (backbone only)
        - Qwen2.5-VL, Qwen3-VL (backbone + visual)

    :param backbone_sim: QuantizationSimModel wrapping backbone.onnx.
    :param visual_sim: Optional QuantizationSimModel wrapping visual.onnx (VLM only).
    :param embedding: Optional ``torch.Tensor`` of shape ``[vocab, hidden]`` loaded
        from ``embedding.pth`` (VLM only). Rotated in-place with R_L.
    :param rotations: Sequence of rotation passes to run. Defaults to
        ``[R1RotationPass()]``. Future R2/R3 passes can be appended here.
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
    if rotations is None:
        rotations = (R1RotationPass(),)

    ctx = _build_context(backbone_sim, visual_sim, embedding)

    # Validate every pass before mutating anything: a bad config must not
    # leave the model half-rotated.
    for rotation in rotations:
        rotation.validate(ctx)

    for rotation in rotations:
        _logger.info("Applying %s rotation pass.", rotation.name)
        rotation.apply(ctx)

    backbone_sim._rebuild_session()  # pylint: disable=protected-access
    if visual_sim is not None:
        visual_sim._rebuild_session()  # pylint: disable=protected-access


def _build_context(
    backbone_sim: QuantizationSimModel,
    visual_sim: Optional[QuantizationSimModel],
    embedding: Optional[torch.Tensor],
) -> SpinquantContext:
    """Run model analysis once and build the context shared across passes."""
    bb_model = backbone_sim.model.model
    bb_cg = backbone_sim.connected_graph
    boundaries, active_norms = get_decoder_block_boundaries(bb_model, bb_cg)
    role_map = get_decoder_role_map(bb_cg, boundaries, active_norms)
    hidden_size = infer_hidden_size(bb_model, role_map)

    visual_merger_linear2 = None
    if visual_sim is not None:
        visual_merger_linear2 = find_merger_linear2(visual_sim.connected_graph)

    _check_embedding_consistency(role_map, embedding)

    return SpinquantContext(
        backbone_sim=backbone_sim,
        backbone_role_map=role_map,
        backbone_active_norms=active_norms,
        backbone_hidden_size=hidden_size,
        visual_sim=visual_sim,
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
