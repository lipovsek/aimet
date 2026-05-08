# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for instantiating a Generator object in onnx/test_genai.py"""

import contextlib

import torch

from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator

from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import TorchONNXInterface


class _VisualONNXAdapter(torch.nn.Module):
    """Reassembles flat ONNX visual outputs into the list structure expected
    by the generator (e.g. deepstack_visual_embeds as a list of tensors)."""

    def __init__(self, interface: TorchONNXInterface, num_list_outputs: int):
        super().__init__()
        self.interface = interface
        self.num_list_outputs = num_list_outputs

    @property
    def config(self):
        return self.interface.config

    @property
    def device(self):
        return self.interface.device

    @property
    def dtype(self):
        return self.interface.dtype

    def forward(self, *args, **kwargs):
        outputs = self.interface(*args, **kwargs)
        if self.num_list_outputs > 0 and isinstance(outputs, (list, tuple)):
            base = list(outputs[: -self.num_list_outputs])
            tail = list(outputs[-self.num_list_outputs :])
            return tuple(base + [tail])
        return outputs


class ONNXFPModeMixin:
    """Mixin that provides fp_mode() for ONNX QuantSim generators."""

    @contextlib.contextmanager
    def fp_mode(self):
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    self.sim_collection.backbone._remove_quantization_nodes()
                )
                self.sim_collection.backbone._rebuild_session()
                if self.sim_collection.visual is not None:
                    stack.enter_context(
                        self.sim_collection.visual._remove_quantization_nodes()
                    )
                    self.sim_collection.visual._rebuild_session()
                yield
        finally:
            self.sim_collection.backbone._rebuild_session()
            if self.sim_collection.visual is not None:
                self.sim_collection.visual._rebuild_session()


class ONNXDevicePlacementMixin:
    """Mixin that provides on_device() for ONNX QuantSim generators.

    Swaps execution providers and rebuilds sessions to move inference to the
    target device. Currently supports moving from CUDA to CPU.
    """

    @contextlib.contextmanager
    def on_device(self, device):
        device = torch.device(device)
        sims = [self.sim_collection.backbone]
        if self.sim_collection.visual is not None:
            sims.append(self.sim_collection.visual)

        original_providers = [sim.providers for sim in sims]
        target_providers = _providers_for_device(device)

        if all(
            _current_device_type(providers) == device.type
            for providers in original_providers
        ):
            yield
            return

        try:
            for sim in sims:
                sim.providers = target_providers
                sim._rebuild_session()
            yield
        finally:
            for sim, orig in zip(sims, original_providers):
                sim.providers = orig
                sim._rebuild_session()


def _current_device_type(providers) -> str:
    for p in providers:
        name = p if isinstance(p, str) else p[0]
        if name == "CUDAExecutionProvider":
            return "cuda"
    return "cpu"


def _providers_for_device(device: torch.device):
    if device.type == "cuda":
        if device.index is not None:
            return [
                ("CUDAExecutionProvider", {"device_id": device.index}),
                "CPUExecutionProvider",
            ]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def generator_factory(
    sim_collection: SimCollection,
    generator_cls: type[Generator],
    tokenizer,
    sequence_length,
    context_length,
    visual_output_names=None,
    **model_kwargs,
) -> Generator:
    # Compose the generator class with ONNX-specific mixins
    mixed_cls = type(
        generator_cls.__name__,
        (ONNXFPModeMixin, ONNXDevicePlacementMixin, generator_cls),
        {},
    )

    if sim_collection.is_vlm():
        assert issubclass(generator_cls, VLM_Generator)
        vision_interface = TorchONNXInterface(
            sim_collection.visual, sim_collection.config
        )
        # Wrap vision interface to reassemble list outputs (e.g. deepstack)
        vis_cfg = getattr(sim_collection.config, "vision_config", None)
        ds_indexes = getattr(vis_cfg, "deepstack_visual_indexes", None)
        if ds_indexes:
            vision_interface = _VisualONNXAdapter(
                vision_interface, num_list_outputs=len(ds_indexes)
            )
        return mixed_cls(
            backbone_model=TorchONNXInterface(
                sim_collection.backbone, sim_collection.config.text_config
            ),
            vision_model=vision_interface,
            embedding=sim_collection.embedding,
            tokenizer=tokenizer,
            position_id_processor=sim_collection.position_id_processor,
            sequence_length=sequence_length,
            context_length=context_length,
            config=sim_collection.config,
            visual_output_names=visual_output_names,
            sim_collection=sim_collection,
            **model_kwargs,
        )
    else:
        return mixed_cls(
            model=TorchONNXInterface(sim_collection.backbone, sim_collection.config),
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            sim_collection=sim_collection,
            **model_kwargs,
        )
