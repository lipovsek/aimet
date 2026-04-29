# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Precision configuration for GenAI model quantization."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from enum import Enum

try:
    from aimet_onnx.common.defs import (
        QTYPE_ALIASES,
        qtype,
        int2,
        int4,
        int8,
        int16,
        float16,
        float32,
    )
except ImportError:
    from aimet_torch.common.defs import (
        QTYPE_ALIASES,
        qtype,
        int2,
        int4,
        int8,
        int16,
        float16,
        float32,
    )


class Granularity(Enum):
    PCQ = "PCQ"
    BQ = "BQ"
    LPBQ = "LPBQ"


def resolve_qtype(value: int | str | qtype) -> qtype:
    """Convert an int bitwidth or string alias to a qtype."""
    if isinstance(value, qtype):
        return value
    if isinstance(value, int):
        return qtype.int(value)
    if value in QTYPE_ALIASES:
        return QTYPE_ALIASES[value]
    raise ValueError(f"Unsupported quantization dtype: {value}")


@dataclass
class WeightPrecision:
    """Precision settings for a weight tensor group (blocks, lm_head, embedding, visual)."""

    qtype: qtype = int4
    granularity: Granularity = Granularity.PCQ
    block_size: int | None = None

    def __post_init__(self):
        if isinstance(self.granularity, str):
            try:
                self.granularity = Granularity(self.granularity)
            except ValueError:
                raise ValueError(
                    f"Invalid granularity '{self.granularity}'. "
                    f"Must be one of: {', '.join(g.value for g in Granularity)}"
                )
        if self.granularity in (Granularity.BQ, Granularity.LPBQ):
            assert self.block_size is not None, (
                f"block_size is required for {self.granularity.value} granularity."
            )

    def to_dict(self) -> dict:
        d = {
            "qtype": repr(self.qtype),
            "granularity": self.granularity.value,
        }
        if self.block_size is not None:
            d["block_size"] = self.block_size
        return d

    @classmethod
    def from_dict(cls, d: dict | str | int | None, **defaults) -> WeightPrecision:
        """Parse a weight precision dict, falling back to keyword defaults."""
        if d is None:
            return cls(**defaults)

        if isinstance(d, int):
            d = {"qtype": d}
        merged = {**defaults, **d}
        if "qtype" in merged:
            merged["qtype"] = resolve_qtype(merged["qtype"])
            if merged["qtype"] in (float16, float32):
                raise ValueError("Floating-point weight precision is not supported.")

        return cls(
            **{
                k: v
                for k, v in merged.items()
                if k in ("qtype", "granularity", "block_size")
            }
        )


@dataclass
class PrecisionConfig:
    """Centralized precision configuration parsed from the YAML ``precision:`` section.

    When the section is omitted, defaults match existing hardcoded values:
    W4A16, lm_head W8 PCQ, KV cache 8-bit.
    """

    activations: qtype = int16
    kv_cache: qtype = int8
    embedding: qtype = int16
    lm_head: WeightPrecision = field(
        default_factory=lambda: WeightPrecision(qtype=int8)
    )
    blocks: dict[str, WeightPrecision] = field(
        default_factory=lambda: {"default": WeightPrecision(qtype=int4)}
    )
    visual_weight: WeightPrecision | None = None
    visual_activations: qtype | None = None

    def ensure_visual_defaults(self) -> None:
        """Populate visual precision fields with defaults if not already set.

        Called when the model is known to be a VLM so that visual precision
        is always explicitly recorded rather than silently falling back to
        backbone settings.
        """
        if self.visual_weight is None:
            self.visual_weight = WeightPrecision(qtype=int8)
        if self.visual_activations is None:
            self.visual_activations = int16

    def weight_identity(self) -> dict:
        """Return the precision fields that affect weight-modifying recipes.

        Excludes activations, kv_cache, and visual_activations since cacheable
        recipes (SpinQuant, AdaScale, SeqMSE) only modify weights and weight
        encodings, not activation quantizers.
        """
        d = {
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
            "lm_head": self.lm_head.to_dict(),
            "embedding": repr(self.embedding),
        }
        if self.visual_weight is not None:
            d["visual_weight"] = self.visual_weight.to_dict()
        return d

    def to_dict(self) -> dict:
        d = {
            "activations": repr(self.activations),
            "kv_cache": repr(self.resolve_kv_cache_qtype()),
            "embedding": repr(self.resolve_embedding_qtype()),
            "lm_head": self.lm_head.to_dict(),
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
        }
        if self.visual_weight is not None:
            d["visual"] = {
                "weight": self.visual_weight.to_dict(),
                "activations": repr(self.visual_activations)
                if self.visual_activations
                else None,
            }
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> PrecisionConfig:
        """Parse the ``precision:`` section of a YAML config.

        Returns the default config when *d* is ``None``.
        """
        if d is None:
            return cls()

        kwargs: dict = {}

        if "activations" in d:
            kwargs["activations"] = resolve_qtype(d["activations"])

        if "kv_cache" in d:
            kwargs["kv_cache"] = resolve_qtype(d["kv_cache"])

        if "lm_head" in d:
            kwargs["lm_head"] = WeightPrecision.from_dict(d["lm_head"], qtype=int8)

        if "embedding" in d:
            kwargs["embedding"] = resolve_qtype(d["embedding"])

        if "blocks" in d:
            blocks_raw = d["blocks"]
            blocks = {}

            # If blocks_raw is an int/str or a flat WeightPrecision dict,
            # treat the whole value as the "default" block config.
            _wp_keys = {field.name for field in fields(WeightPrecision)}
            if isinstance(blocks_raw, (int, str)):
                blocks["default"] = WeightPrecision.from_dict(blocks_raw, qtype=int4)
            elif isinstance(blocks_raw, dict) and blocks_raw.keys() <= _wp_keys:
                blocks["default"] = WeightPrecision.from_dict(blocks_raw, qtype=int4)
            else:
                for key, value in blocks_raw.items():
                    if key == "default":
                        blocks[key] = WeightPrecision.from_dict(value, qtype=int4)
                    else:
                        raise ValueError(
                            f"Per-block-range precision (key '{key}') is not yet supported. "
                            f"Only 'default' is currently accepted under precision.blocks."
                        )
            kwargs["blocks"] = blocks

        if "visual" in d:
            visual = d["visual"]
            if "weight" in visual:
                kwargs["visual_weight"] = WeightPrecision.from_dict(
                    visual["weight"], qtype=int8
                )
            else:
                kwargs["visual_weight"] = WeightPrecision(qtype=int8)
            kwargs["visual_activations"] = resolve_qtype(visual.get("activations", 16))

        return cls(**kwargs)

    def resolve_kv_cache_qtype(self, value: int | str | None = None) -> qtype:
        """Convert an kv_cache_bits value to an int bitwidth or qtype.
        When *value* is ``None``, uses ``self.kv_cache_bits``.
        """
        if self.activations in (float16, float32):
            print(
                "Warning: activation_bits is set to a floating-point type, so KV cache will be ignored and set to the same floating-point type."
            )
            return self.activations
        return value or self.kv_cache

    def resolve_embedding_qtype(self, value: int | str | None = None) -> qtype:
        if self.activations in (float16, float32):
            print(
                "Warning: activation_bits is set to a floating-point type, so embedding will be ignored and set to the same floating-point type."
            )
            return self.activations
        return value or self.embedding
