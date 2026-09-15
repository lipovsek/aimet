# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level SpinQuant API for ONNX QuantizationSimModel.

This module is the entry-point orchestrator. It performs model analysis once,
builds a :class:`SpinquantContext`, then runs the rotation passes selected by
the caller via boolean flags (``enable_r1`` / ``enable_r2``).
"""

from typing import List, Optional

import onnx
import onnx_ir
import torch

from aimet_onnx.common.utils import AimetLogger

from aimet_onnx.experimental.llm_topology.ir_adapter import resolve_topology
from aimet_onnx.experimental.llm_topology.ir_analysis import build_analysis_ir
from aimet_onnx.experimental.llm_topology.topology import (
    analyze_llm_topology,
)
from aimet_onnx.experimental.spinquant.model_analysis import (
    find_merger_linear2,
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
    4. Serializing the result back onto the caller's ``ModelProto``\\ s.

    The rotations are performed on an ``onnx_ir`` copy of the graph and written
    back only once every pass has succeeded, so a failure part-way through leaves
    the caller's model untouched rather than half-rotated. (The ``embedding``
    tensor is the exception: it is a ``torch.Tensor`` rotated in place.)

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

    # The passes rewrote the IR; serialize it back onto the caller's proto(s).
    # Doing this only once every pass has succeeded is what makes a mid-flight
    # failure leave the caller's model untouched rather than half-rotated.
    _write_back(model, ctx.backbone_ir)
    if visual_model is not None:
        _write_back(visual_model, ctx.visual_ir)


def _build_context(
    model: onnx.ModelProto,
    visual_model: Optional[onnx.ModelProto],
    embedding: Optional[torch.Tensor],
) -> SpinquantContext:
    """Run model analysis once and build the context shared across passes.

    Builds two IRs for the backbone — the faithful one the passes mutate, and the
    quantizer-stripped / RMSNorm-fused analysis one the detection runs on — and
    resolves the analyzed topology onto the faithful one. See
    :class:`SpinquantContext` on why the two cannot be the same object.
    """
    # A faithful copy of the caller's graph: not sorted, not stripped, not fused,
    # so what we hand back differs from what we were given only where a rotation
    # actually changed something.
    backbone_ir = onnx_ir.from_proto(model)
    analysis_ir = build_analysis_ir(model)

    # Derives block boundaries, per-block roles, active norms, hidden_size and
    # head_dim in one pass. head_dim is only needed by R2/R3; it is left None
    # when the export has no KV-cache 'past_value' input, and those passes raise
    # a targeted error when they actually need it.
    name_topology = analyze_llm_topology(model, ir_model=analysis_ir)
    topology = resolve_topology(name_topology, backbone_ir)

    visual_ir = None
    visual_merger_linear2 = None
    if visual_model is not None:
        visual_ir = onnx_ir.from_proto(visual_model)
        visual_merger_linear2 = find_merger_linear2(visual_ir)

    _check_embedding_consistency(topology, embedding)

    return SpinquantContext(
        backbone_ir=backbone_ir,
        backbone_analysis_ir=analysis_ir,
        backbone_topology=topology,
        backbone_active_norms=topology.active_norms,
        backbone_hidden_size=topology.hidden_size,
        backbone_head_dim=topology.head_dim,
        visual_ir=visual_ir,
        visual_merger_linear2=visual_merger_linear2,
        embedding=embedding,
    )


def _write_back(model: onnx.ModelProto, ir_model: onnx_ir.Model) -> None:
    """Serialize ``ir_model`` onto ``model`` in place.

    Every model the caller handed us is written back, whether or not the enabled
    passes touched it. Tracking which IRs were mutated would save a serialization
    in the rare configuration that passes a visual encoder without enabling R1,
    at the cost of silently dropping rotations the day a pass forgets to report
    one — an unnecessary round trip is the cheaper mistake.
    """
    model.CopyFrom(onnx_ir.to_proto(ir_model))


def _check_embedding_consistency(topology, embedding: Optional[torch.Tensor]) -> None:
    """Reject inconsistent (embedding, embed_tokens) combinations.

    A backbone with embed_tokens must not receive an external embedding: R1 is
    already absorbed by the Gather weight, so the tensor would be rotated twice.
    """
    if embedding is not None and topology.embed_tokens:
        raise ValueError(
            "embedding was provided but backbone contains embed_tokens op(s). "
            "Pass embedding only for VLM backbones exported with use_inputs_embeds=True "
            "(i.e. backbone has no Gather op for token embeddings)."
        )
