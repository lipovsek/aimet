# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-layer cache descriptors for models with heterogeneous attention types."""

from dataclasses import dataclass
from enum import Enum

from transformers import PretrainedConfig


class AttentionType(Enum):
    """Attention mechanism used by a decoder layer."""

    FULL = "full_attention"
    SLIDING_WINDOW = "sliding_attention"
    LINEAR = "linear_attention"


# Mapping from HuggingFace config ``layer_types`` strings to AttentionType
_HF_LAYER_TYPE_MAP: dict[str, AttentionType] = {
    "full_attention": AttentionType.FULL,
    "sliding_attention": AttentionType.SLIDING_WINDOW,
    "linear_attention": AttentionType.LINEAR,
    "recurrent": AttentionType.LINEAR,
}


@dataclass
class LayerCacheDescriptor:
    """Describes the cache/state requirements of a single decoder layer."""

    layer_idx: int
    attention_type: AttentionType
    num_kv_heads: int
    head_dim: int
    sliding_window_size: int | None = None

    def dummy_state_shape(
        self, batch_size: int, context_length: int, sequence_length: int
    ) -> tuple[int, ...]:
        """Shape of the dummy state tensor for this layer during prepare_inputs.

        Full and sliding_window layers are padded to the full context length so
        that a single 4D attention mask can be applied across all layers.
        """
        if self.attention_type in (AttentionType.FULL, AttentionType.SLIDING_WINDOW):
            return (
                batch_size,
                self.num_kv_heads,
                context_length - sequence_length,
                self.head_dim,
            )
        if self.attention_type == AttentionType.LINEAR:
            return (batch_size, self.num_kv_heads, self.head_dim, self.head_dim)
        raise ValueError(f"Unknown attention type: {self.attention_type}")

    def clip_length(self, max_length: int) -> int | None:
        """Maximum number of KV entries to retain between generation steps.

        Returns *None* for linear attention (state is replaced, not clipped).
        """
        if self.attention_type == AttentionType.FULL:
            return max_length
        if self.attention_type == AttentionType.SLIDING_WINDOW:
            return min(self.sliding_window_size, max_length)
        if self.attention_type == AttentionType.LINEAR:
            return None
        raise ValueError(f"Unknown attention type: {self.attention_type}")


def build_layer_cache_descriptors(
    config: PretrainedConfig,
) -> list[LayerCacheDescriptor]:
    """Build per-layer cache descriptors from a HuggingFace model config.

    Inspects ``config.layer_types``, ``config.sliding_window``, and
    ``config.sliding_window_pattern`` to determine each layer's cache type.
    """
    num_layers = config.num_hidden_layers
    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim") and config.head_dim is not None
        else config.hidden_size // config.num_attention_heads
    )
    num_kv_heads = config.num_key_value_heads
    sliding_window = getattr(config, "sliding_window", None)
    layer_types = getattr(config, "layer_types", None)

    descriptors: list[LayerCacheDescriptor] = []
    for i in range(num_layers):
        hf_layer_type = layer_types[i] if layer_types else None
        mapped = _HF_LAYER_TYPE_MAP.get(hf_layer_type) if hf_layer_type else None

        if mapped is not None:
            attention_type = mapped
        elif _is_sliding_window_layer(config, i):
            attention_type = AttentionType.SLIDING_WINDOW
        else:
            attention_type = AttentionType.FULL

        sw_size = (
            sliding_window if attention_type == AttentionType.SLIDING_WINDOW else None
        )

        descriptors.append(
            LayerCacheDescriptor(
                layer_idx=i,
                attention_type=attention_type,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                sliding_window_size=sw_size,
            )
        )

    return descriptors


def _is_sliding_window_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    """Determine whether *layer_idx* uses sliding window attention.

    Detection heuristics (checked in order):
    1. ``config.sliding_window_pattern`` — modular pattern (e.g. every Nth layer).
    2. Known model types with hardcoded patterns (e.g. Gemma 2 alternates).
    3. If ``config.sliding_window`` is set but no pattern is found, all layers
       are assumed to use sliding window.
    """
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        return False

    # Explicit pattern from config
    pattern = getattr(config, "sliding_window_pattern", None)
    if pattern is not None:
        return (layer_idx % pattern) == 0

    # Gemma 2: odd-indexed layers use sliding window
    model_type = getattr(config, "model_type", "")
    if model_type == "gemma2":
        return layer_idx % 2 != 0

    # Fallback: sliding_window is set with no pattern — assume no layers
    return False
