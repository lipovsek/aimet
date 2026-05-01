# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for building GenAI models"""

import torch
from transformers import PreTrainedModel, DynamicCache

from GenAILab.qai_hub_lm.utils.layer_cache import _resolve_text_config


def _patch_sdpa_mask():
    # In transformers >=5.3.0, _preprocess_mask_arguments derives q_length from
    # inputs_embeds.shape[1], which under torch.jit.trace yields a 0-dim tensor
    # instead of a Python int.
    # Fix: convert 0-dim tensors to int.
    try:
        import transformers.masking_utils as _mu

        _orig = _mu.sdpa_mask

        def _patched(batch_size, q_length, *args, **kwargs):
            if isinstance(q_length, torch.Tensor) and q_length.ndim == 0:
                q_length = q_length.item()
            return _orig(batch_size, q_length, *args, **kwargs)

        _mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = _patched
    except (ImportError, AttributeError):
        pass


# TODO: Remove this patch once the fix applied in transformers itself.
_patch_sdpa_mask()


def compute_vision_input_shapes(
    image_size: tuple[int, int],
    vision_config,
) -> tuple[int, int, int, int]:
    """Compute vision encoder input shapes from a target image size.

    Args:
        image_size: Target (width, height) that images will be resized to.
            Follows PIL convention.
        vision_config: HF vision config with ``patch_size``,
            ``spatial_merge_size``, ``temporal_patch_size``, and
            ``in_channels`` attributes.

    Returns:
        (num_patches, pixel_dim, h_patches, w_patches)
    """
    w, h = image_size
    patch_size = vision_config.patch_size
    merge_size = vision_config.spatial_merge_size
    temporal_patch_size = vision_config.temporal_patch_size
    in_channels = vision_config.in_channels

    # The HF processor rounds image dimensions down to the nearest
    # multiple of (patch_size * spatial_merge_size).
    factor = patch_size * merge_size
    h_patches = (h // factor) * merge_size
    w_patches = (w // factor) * merge_size

    num_patches = h_patches * w_patches
    pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size

    return num_patches, pixel_dim, h_patches, w_patches


class PositionIdContext:
    """Minimal stand-in for ``self`` when calling HF's unbound ``get_rope_index``.

    HF's ``get_rope_index`` is an instance method that accesses ``self.config``
    and may call sibling methods (e.g. ``self.get_vision_position_ids``).  In our
    framework the position-ID computation must work without a real HF model
    instance (e.g. in the ONNX path).  This proxy satisfies the ``self`` contract
    by holding the config and delegating any other attribute lookups to the
    original HF model *class* (bound to this proxy).
    """

    def __init__(self, config, model_cls):
        self.config = config
        self._model_cls = model_cls

    def __getattr__(self, name):
        attr = getattr(self._model_cls, name)
        if callable(attr):
            return attr.__get__(self, type(self))
        return attr


class ONNXExportableModuleWithCache(torch.nn.Module):
    """
    Helper class to enable Torch JIT trace and ONNX export of HuggingFace models
    that produce and consume Cache objects. Supports both LLM and VLM backbones.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        lm_head: torch.nn.Module | None = None,
        use_inputs_embeds: bool = False,
        extra_input_names: tuple[str, ...] = (),
        cache_type: type = DynamicCache,
    ):
        """
        :param model: The HuggingFace model to wrap
        :param lm_head: Optional LM head (for VLM backbones where head is separate)
        :param use_inputs_embeds: If True, first input is inputs_embeds; else input_ids
        :param extra_input_names: Names of additional inputs to pass through to model
        :param cache_type: Cache class to construct from flattened KV pairs.
            Defaults to ``DynamicCache``. Models with hybrid attention (e.g.
            mixing full attention with linear/recurrent layers) can pass
            ``HybridCache`` or another cache class.
        """
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.use_inputs_embeds = use_inputs_embeds
        self.extra_input_names = extra_input_names
        self.cache_type = cache_type

    @property
    def device(self):
        """Return model device"""
        return self.model.device

    @property
    def dtype(self):
        """Return model dtype"""
        return self.model.dtype

    @property
    def config(self):
        """Return model config"""
        return self.model.config

    def _build_cache(self, past_key_values: tuple[tuple[torch.Tensor, ...], ...]):
        """Build a cache object from flattened state pairs using ``self.cache_type``."""
        # Avoid passing config to DynamicCache — it creates DynamicSlidingWindowLayer
        # for sliding-window layers, which clips KV entries internally. Our 4D attention
        # mask already handles the windowing semantics, so we need uniform-sized caches.
        if self.cache_type is DynamicCache:
            kv_cache = DynamicCache()
        else:
            kv_cache = self.cache_type(config=self.config)
        layer_types = getattr(self.config, "layer_types", None)
        for layer_idx, (state_a, state_b) in enumerate(
            zip(past_key_values[::2], past_key_values[1::2])
        ):
            if layer_types and layer_types[layer_idx] == "linear_attention":
                # Linear attention layers use conv_states / recurrent_states
                kv_cache.conv_states[layer_idx] = state_a
                kv_cache.recurrent_states[layer_idx] = state_b
            else:
                kv_cache.update(state_a, state_b, layer_idx, {})
        return kv_cache

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        input_or_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        *args: tuple[torch.Tensor, ...],
    ):
        """
        Redefine model forward to convert to/from Huggingface DynamicCache objects.

        Args layout: [input_or_embeds, attention_mask, position_ids, *kv_cache_pairs, *extra_inputs]

        The number of extra inputs is determined by len(self.extra_input_names).
        KV cache pairs come before extra inputs in *args.
        """
        num_kv = 2 * _resolve_text_config(self.model.config).num_hidden_layers
        num_extra = len(self.extra_input_names)
        expected = num_kv + num_extra
        if len(args) != 0 and len(args) < num_kv:
            raise RuntimeError(
                f"Expected at least {num_kv} args (KV pairs) but got {len(args)}."
            )

        # Split args into KV cache and extra inputs
        past_key_values = args[:num_kv]
        if num_extra > 0 and len(args) >= expected:
            extra_inputs = args[num_kv : num_kv + num_extra]
            extra_kwargs = dict(zip(self.extra_input_names, extra_inputs))
        else:
            extra_kwargs = {}

        # Build cache from flattened state pairs
        kv_cache = self._build_cache(past_key_values)

        # Build model kwargs
        model_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": kv_cache,
            "num_logits_to_return": 0,
            "return_dict": False,
            **extra_kwargs,
        }

        if self.use_inputs_embeds:
            model_kwargs["input_ids"] = None
            model_kwargs["inputs_embeds"] = input_or_embeds
        else:
            model_kwargs["input_ids"] = input_or_embeds

        # Call underlying model
        outputs = self.model(**model_kwargs)
        hidden_states_or_logits, new_past_key_values = outputs[0], outputs[1]

        # Apply lm_head if provided (VLM backbone case)
        if self.lm_head is not None:
            lm_logits = self.lm_head(hidden_states_or_logits)
        else:
            lm_logits = hidden_states_or_logits

        # Flatten output KV cache
        flat_output_past_key_values = []
        layer_types = getattr(self.config, "layer_types", None)
        for layer in range(len(new_past_key_values)):
            if layer_types and layer_types[layer] == "linear_attention":
                # Linear attention: extract conv_state and recurrent_state
                flat_output_past_key_values.append(
                    new_past_key_values.conv_states[layer]
                )
                flat_output_past_key_values.append(
                    new_past_key_values.recurrent_states[layer]
                )
            elif hasattr(new_past_key_values, "value_cache"):
                keys = new_past_key_values.key_cache[layer]
                values = new_past_key_values.value_cache[layer]
                flat_output_past_key_values += [keys, values]
            elif hasattr(new_past_key_values.layers[layer], "keys"):
                keys = new_past_key_values.layers[layer].keys
                values = new_past_key_values.layers[layer].values
                flat_output_past_key_values += [keys, values]
            else:
                keys = new_past_key_values.layers[layer][0]
                values = new_past_key_values.layers[layer][1]
                flat_output_past_key_values += [keys, values]

        return lm_logits, *flat_output_past_key_values
