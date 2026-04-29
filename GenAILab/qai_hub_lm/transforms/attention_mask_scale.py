# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-layer attention mask scaling adaptation.

Registers a forward pre-hook on each specified decoder layer that multiplies
the ``attention_mask`` by a configurable scalar.  This allows certain layers
to see a stronger (more negative) mask than the global ``attention_mask_min``
without changing the value for the rest of the model.

YAML usage::

    model:
      model_id: meta-llama/Llama-3.2-1B
      attention_mask_min: -100
      adaptations:
        - AttentionMaskScale:
            layer_multipliers:
              0: 10.0
              5: 25.0
"""

import torch
from transformers import PreTrainedModel

from GenAILab.bench.yaml_config_parser import YAMLConfigParser


def _get_decoder_layers(hf_model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the decoder ``ModuleList`` from a HuggingFace model.

    Handles both top-level decoders (``model.layers``) and nested ones
    (``model.model.layers``).
    """
    hf_model = getattr(hf_model, "model", hf_model)
    language_model = getattr(hf_model, "language_model", hf_model)
    if hasattr(language_model, "layers"):
        return language_model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(language_model).__name__}. "
        "Expected a `.layers` or `.model.layers` attribute."
    )


def _mask_multiplier_hook(module, args, kwargs):
    """Forward pre-hook that scales ``attention_mask`` by the layer's multiplier."""
    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
        multiplier = module._attn_mask_multiplier
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = kwargs["attention_mask"] * multiplier
    return args, kwargs


def _register_mask_multiplier(layer: torch.nn.Module, multiplier: float):
    """Attach a mask-scaling pre-hook to a single decoder layer."""
    layer._attn_mask_multiplier = multiplier
    layer.register_forward_pre_hook(_mask_multiplier_hook, with_kwargs=True)


@YAMLConfigParser.register_adaptation("AttentionMaskScale", model_type="*")
class AttentionMaskScaleAdaptation:
    """Mixin that applies per-layer attention mask multipliers after model load.

    Expects ``layer_multipliers`` as a class attribute (set automatically by
    ``YAMLConfigParser.get_model_class`` from the YAML config).
    """

    layer_multipliers: dict = {}

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        model = super().instantiate_model(*args, **kwargs)

        layer_multipliers = getattr(cls, "layer_multipliers", {})
        if layer_multipliers:
            decoder_layers = _get_decoder_layers(model)
            for layer_idx, multiplier in layer_multipliers.items():
                _register_mask_multiplier(
                    decoder_layers[int(layer_idx)], float(multiplier)
                )

        return model
