# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Base class and shared context for SpinQuant rotation passes."""

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from aimet_onnx.meta.operations import Op
from aimet_onnx.quantsim import QuantizationSimModel

from aimet_onnx.experimental.spinquant.model_analysis import (
    ActiveNorm,
    DecoderModelRoleMap,
)


@dataclass
class SpinquantContext:
    """Inputs and pre-computed analysis shared across rotation passes.

    Built once by :func:`apply_spinquant`, then handed to every pass. Passes
    must not mutate the analysis fields; they may freely mutate the underlying
    ONNX model via ``backbone_sim`` / ``visual_sim``.

    :param backbone_sim: QuantizationSimModel wrapping backbone.onnx.
    :param backbone_role_map: Decoder role map for the backbone.
    :param backbone_active_norms: Active norms in topological order.
    :param backbone_hidden_size: Hidden dimension of the language backbone residual stream.
    :param visual_sim: Optional QuantizationSimModel wrapping visual.onnx (VLM only).
    :param visual_merger_linear2: PatchMerger linear_fc2 ops (VLM only).
    :param embedding: Optional raw embedding tensor (VLM with use_inputs_embeds=True).
    """

    backbone_sim: QuantizationSimModel
    backbone_role_map: DecoderModelRoleMap
    backbone_active_norms: List[ActiveNorm]
    backbone_hidden_size: int
    visual_sim: Optional[QuantizationSimModel] = None
    visual_merger_linear2: Optional[List[Op]] = field(default=None)
    embedding: Optional[torch.Tensor] = None


class RotationPass(abc.ABC):
    """A single SpinQuant rotation (e.g. R1, R2, R3).

    Sub-classes encapsulate:

    * which rotation matrix to construct, and at what dimension;
    * which ops in the role map to rotate, and on which axis;
    * any architectural pre-conditions to check before mutating the model;
    * any required setup (e.g. norm fusion) that must run before the rotation.

    The orchestrator validates every pass first, then applies them in order, so
    a bad configuration cannot leave the model half-rotated.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier used in log messages."""

    @abc.abstractmethod
    def validate(self, ctx: SpinquantContext) -> None:
        """Raise if the model or context is incompatible with this rotation.

        Called for every pass before any pass mutates the model.
        """

    @abc.abstractmethod
    def apply(self, ctx: SpinquantContext) -> None:
        """Mutate the ONNX model(s) and any auxiliary tensors in-place."""
