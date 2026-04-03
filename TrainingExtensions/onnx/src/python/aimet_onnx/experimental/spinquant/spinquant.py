# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level SpinQuant API for ONNX QuantizationSimModel."""

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.experimental.spinquant.apply_rotation import (
    apply_r1_rotation,
    _infer_hidden_size,
    _validate_all_weights,
)
from aimet_onnx.experimental.spinquant.block_identifier import (
    get_decoder_block_boundaries,
    get_decoder_role_map,
)
from aimet_onnx.experimental.spinquant.fuse_norm import fuse_norm_layers_into_linears

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SpinQuant)


def apply_spinquant(
    sim: QuantizationSimModel,
) -> None:
    """Apply SpinQuant rotation transforms to an ONNX transformer model.

    SpinQuant applies orthogonal Hadamard rotations to model weights to reduce
    quantization error. This function modifies the QuantizationSimModel in-place by:

    1. Fusing RMS normalization layers into subsequent linear layers
    2. Applying R1 Hadamard rotations to embeddings, attention, and MLP layers

    Must be called BEFORE sim.compute_encodings(). The rotation modifies float
    weight initializers; compute_encodings must run afterward to calibrate
    quantizer scales on the rotated weights.

    Supported architectures:
        - LLaMA
        - Qwen2, Qwen3
        - Phi3

    :param sim: A QuantizationSimModel wrapping an ONNX transformer model. The
        model must have untied embed_tokens and lm_head weights.
    :raises ValueError: If block detection or role classification fails.

    Example::

        import onnx
        from aimet_onnx.quantsim import QuantizationSimModel
        from aimet_onnx.experimental.spinquant import apply_spinquant

        model = onnx.load("llama.onnx")
        sim = QuantizationSimModel(model)
        apply_spinquant(sim)
        sim.compute_encodings(calibration_data)
        sim.export("output_dir", "llama_spinquant")
    """
    model = sim.model.model
    cg = sim.connected_graph

    block_boundaries, active_norms = get_decoder_block_boundaries(model, cg)
    role_map = get_decoder_role_map(cg, block_boundaries, active_norms)
    _logger.info(
        "Detected %d decoder block(s), %d embed_tokens op(s), %d lm_head op(s).",
        len(role_map.blocks),
        len(role_map.embed_tokens),
        len(role_map.lm_head),
    )

    hidden_size = _infer_hidden_size(model, role_map)
    _validate_all_weights(model, role_map, hidden_size)

    fuse_norm_layers_into_linears(model, active_norms)
    apply_r1_rotation(model, role_map, hidden_size)

    sim._rebuild_session()  # pylint: disable=protected-access
    _logger.info(
        "R1 rotation applied successfully. Call sim.compute_encodings() to calibrate quantizer scales."
    )
