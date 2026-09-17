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

from GenAILab.qai_hub_lm.schema.components import (
    MODALITY_COMPONENTS,
    require_component,
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
        # block_size only applies to integer weights; FP weights ignore granularity.
        if self.qtype not in (float16, float32) and self.granularity in (
            Granularity.BQ,
            Granularity.LPBQ,
        ):
            assert self.block_size is not None, (
                f"block_size is required for {self.granularity.value} granularity."
            )

    @property
    def is_float(self) -> bool:
        return self.qtype in (float16, float32)

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
    audio_weight: WeightPrecision | None = None
    audio_activations: qtype | None = None

    # ---- generic per-component access ---------------------------------------
    # Modality-encoder precision is stored as ``<component>_weight`` /
    # ``<component>_activations`` pairs; these accessors let callers stay generic
    # over MODALITY_COMPONENTS.

    def component_weight(self, component: str) -> WeightPrecision | None:
        require_component(component)
        return getattr(self, f"{component}_weight")

    def component_activations(self, component: str) -> qtype | None:
        require_component(component)
        return getattr(self, f"{component}_activations")

    @classmethod
    def from_schema(cls, schema) -> "PrecisionConfig":
        """Resolve a validated ``schema.PrecisionSchema`` into a PrecisionConfig.

        This is the GenAILab-side resolution step: the schema (clean, aimet-free,
        synced to AIHM) carries qtypes as named strings / bare ints; here we bind
        them to aimet ``qtype`` objects via ``resolve_qtype``. AIHM resolves the
        same schema into its own representation. Parity with the legacy
        ``from_dict`` path is enforced by ``tests/.../test_precision_parity.py``.
        """

        def _wp(wp_schema) -> WeightPrecision:
            # schema enums are str-enums; .value yields "int4"/"PCQ" etc., which
            # resolve_qtype / Granularity already accept. Bare ints pass through.
            qt = wp_schema.qtype
            return WeightPrecision(
                qtype=resolve_qtype(qt.value if isinstance(qt, Enum) else qt),
                granularity=Granularity(wp_schema.granularity.value),
                block_size=wp_schema.block_size,
            )

        def _qt(value) -> qtype:
            return resolve_qtype(value.value if isinstance(value, Enum) else value)

        kwargs: dict = {
            "activations": _qt(schema.activations),
            "kv_cache": _qt(schema.kv_cache),
            "embedding": _qt(schema.embedding),
            "lm_head": _wp(schema.lm_head),
            "blocks": {k: _wp(v) for k, v in schema.blocks.items()},
        }
        for comp in MODALITY_COMPONENTS:
            block = getattr(schema, comp, None)
            if block is not None:
                kwargs[f"{comp}_weight"] = _wp(block.weight)
                kwargs[f"{comp}_activations"] = _qt(block.activations)
        return cls(**kwargs)

    def ensure_component_defaults(self, component: str) -> None:
        """Populate one component's precision fields with defaults if unset.

        Called when the model is known to have the component, so that its
        precision is always explicitly recorded rather than silently falling
        back to backbone settings.
        """
        require_component(component)
        if getattr(self, f"{component}_weight") is None:
            setattr(self, f"{component}_weight", WeightPrecision(qtype=int8))
        if getattr(self, f"{component}_activations") is None:
            setattr(self, f"{component}_activations", int16)

    def weight_identity(self) -> dict:
        """Return the precision fields that affect weight-modifying recipes.

        Excludes activations, kv_cache, and per-component activations since
        cacheable recipes (SpinQuant, AdaScale, SeqMSE) only modify weights and
        weight encodings, not activation quantizers.

        Keys are emitted in MODALITY_COMPONENTS order and only when set, so a
        config with no audio block hashes byte-identically to before audio
        existed.
        """
        d = {
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
            "lm_head": self.lm_head.to_dict(),
            "embedding": repr(self.embedding),
        }
        for comp in MODALITY_COMPONENTS:
            weight = getattr(self, f"{comp}_weight")
            if weight is not None:
                d[f"{comp}_weight"] = weight.to_dict()
        return d

    def to_dict(self) -> dict:
        d = {
            "activations": repr(self.activations),
            "kv_cache": repr(self.resolve_kv_cache_qtype()),
            "embedding": repr(self.resolve_embedding_qtype()),
            "lm_head": self.lm_head.to_dict(),
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
        }
        for comp in MODALITY_COMPONENTS:
            weight = getattr(self, f"{comp}_weight")
            activations = getattr(self, f"{comp}_activations")
            if weight is not None:
                d[comp] = {
                    "weight": weight.to_dict(),
                    "activations": repr(activations) if activations else None,
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

        for comp in MODALITY_COMPONENTS:
            if comp not in d:
                continue
            block = d[comp]
            if "weight" in block:
                kwargs[f"{comp}_weight"] = WeightPrecision.from_dict(
                    block["weight"], qtype=int8
                )
                if kwargs[f"{comp}_weight"].is_float:
                    raise ValueError(
                        "Floating-point weight precision is not supported for "
                        f"{comp}.weight."
                    )
            else:
                kwargs[f"{comp}_weight"] = WeightPrecision(qtype=int8)
            kwargs[f"{comp}_activations"] = resolve_qtype(block.get("activations", 16))

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
