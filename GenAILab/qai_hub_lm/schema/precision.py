# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative schema for the ``precision`` section of a quant config.

Clean, representation-only: bitwidths are ints / a string enum, NO aimet (each
consumer resolves it to qtype objects; GenAILab does so in the aimet-side
``PrecisionConfig.from_schema``). Unlike recipe specs, DEFAULTS ARE THE CONTRACT
here -- omitted section => W4A16, lm_head W8 PCQ, KV int8, embedding int16 --
mirroring the old ``PrecisionConfig`` dataclass, so existing configs validate
unchanged (confirmed by the cluster parity test).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Granularity(str, Enum):
    # Mirrors aimet-side Granularity (redefined to stay aimet-free; drift-guarded by a test).
    PCQ = "PCQ"
    BQ = "BQ"
    LPBQ = "LPBQ"


class QType(str, Enum):
    # Mirrors aimet's QTYPE_ALIASES keys (drift-guarded by a test). A bare int
    # bitwidth is also accepted in YAML -- see QTypeRef.

    int2 = "int2"
    int4 = "int4"
    int8 = "int8"
    int16 = "int16"
    float16 = "float16"
    float32 = "float32"


# A qtype reference as written in YAML: either a named alias or a bare int
# bitwidth (resolve_qtype on the aimet side accepts both).
QTypeRef = QType | int

_FLOAT_QTYPES = {QType.float16, QType.float32}


def _is_float_qtype(q: QTypeRef) -> bool:
    return q in _FLOAT_QTYPES


class WeightPrecisionSchema(BaseModel, extra="forbid"):
    """Precision for a weight group (blocks / lm_head / visual). A bare int/str is the qtype."""

    qtype: QTypeRef = QType.int4
    granularity: Granularity = Granularity.PCQ
    block_size: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data):
        # ``qtype: int4`` shorthand: a bare int or str is the qtype.
        if isinstance(data, (int, str)):
            return {"qtype": data}
        return data

    @model_validator(mode="after")
    def _block_size_required_for_blocked_int(self) -> "WeightPrecisionSchema":
        # block_size required for BQ/LPBQ on integer weights; float ignores granularity.
        if not _is_float_qtype(self.qtype) and self.granularity in (
            Granularity.BQ,
            Granularity.LPBQ,
        ):
            if self.block_size is None:
                raise ValueError(
                    f"block_size is required for {self.granularity.value} granularity."
                )
        return self

    @property
    def is_float(self) -> bool:
        return _is_float_qtype(self.qtype)


def _default_lm_head() -> WeightPrecisionSchema:
    return WeightPrecisionSchema(qtype=QType.int8)


def _default_blocks() -> dict[str, WeightPrecisionSchema]:
    return {"default": WeightPrecisionSchema(qtype=QType.int4)}


class VisualPrecisionSchema(BaseModel, extra="forbid"):
    """Visual-encoder precision (VLMs only): weight + activations."""

    weight: WeightPrecisionSchema = Field(default_factory=_default_lm_head)
    activations: QTypeRef = QType.int16

    @model_validator(mode="after")
    def _weight_not_float(self) -> "VisualPrecisionSchema":
        # Mirrors from_dict: floating-point visual.weight is unsupported.
        if self.weight.is_float:
            raise ValueError(
                "Floating-point weight precision is not supported for visual.weight."
            )
        return self


class PrecisionSchema(BaseModel, extra="forbid"):
    """The validated ``precision:`` section (defaults = the W4A16 contract, see module doc)."""

    activations: QTypeRef = QType.int16
    kv_cache: QTypeRef = QType.int8
    embedding: QTypeRef = QType.int16
    lm_head: WeightPrecisionSchema = Field(default_factory=_default_lm_head)
    blocks: dict[str, WeightPrecisionSchema] = Field(default_factory=_default_blocks)
    visual: VisualPrecisionSchema | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_blocks(cls, data):
        # blocks may be an int/str shorthand, a flat dict, or {"default": {...}}.
        # Only "default" is supported (per-block-range precision not yet allowed).
        if not isinstance(data, dict) or "blocks" not in data:
            return data
        blocks_raw = data["blocks"]
        _wp_keys = {"qtype", "granularity", "block_size"}
        if isinstance(blocks_raw, (int, str)):
            data = {**data, "blocks": {"default": {"qtype": blocks_raw}}}
        elif isinstance(blocks_raw, dict):
            if blocks_raw and set(blocks_raw) <= _wp_keys:
                # flat WeightPrecision dict -> wrap as the default block
                data = {**data, "blocks": {"default": blocks_raw}}
            else:
                bad = set(blocks_raw) - {"default"}
                if bad:
                    raise ValueError(
                        f"Per-block-range precision (keys {sorted(bad)}) is not yet "
                        f"supported. Only 'default' is accepted under precision.blocks."
                    )
        return data

    @model_validator(mode="after")
    def _blocks_has_default(self) -> "PrecisionSchema":
        if "default" not in self.blocks:
            raise ValueError("precision.blocks must contain a 'default' entry.")
        return self
