# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for instantiating a Generator object in onnx/test_genai.py"""

import contextlib

import torch

from GenAILab.shared.models.base import SimCollection
from GenAILab.shared.models.generator import Generator, VLM_Generator

from GenAILab.onnx.models.utils.torch_onnx_interface import TorchONNXInterface


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


def _build_fp_mode(sim_collection: SimCollection):
    """Build a context manager factory that temporarily removes ONNX quantization nodes."""

    @contextlib.contextmanager
    def fp_mode():
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    sim_collection.backbone._remove_quantization_nodes()
                )
                sim_collection.backbone._rebuild_session()
                if sim_collection.visual is not None:
                    stack.enter_context(
                        sim_collection.visual._remove_quantization_nodes()
                    )
                    sim_collection.visual._rebuild_session()
                yield
        finally:
            sim_collection.backbone._rebuild_session()
            if sim_collection.visual is not None:
                sim_collection.visual._rebuild_session()

    return fp_mode


def generator_factory(
    sim_collection: SimCollection,
    generator_cls: type[Generator],
    tokenizer,
    sequence_length,
    context_length,
    visual_output_names=None,
    **model_kwargs,
) -> Generator:
    fp_mode = _build_fp_mode(sim_collection)

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
        return generator_cls(
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
            fp_mode=fp_mode,
            **model_kwargs,
        )
    else:
        return generator_cls(
            model=TorchONNXInterface(sim_collection.backbone, sim_collection.config),
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            fp_mode=fp_mode,
            **model_kwargs,
        )
