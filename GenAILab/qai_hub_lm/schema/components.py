# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Component names and kinds -- the schema-side half of the component registry.

Only what the schema needs: the names, their canonical order, and their kind.
Stdlib-only, because ``qai_hub_lm/schema/`` is an import island rsync'd verbatim
into AI Hub Models.

The model-side half (hooks, token ids, prefill-only kwargs) is in
``qai_hub_lm/models/components.py``, which imports these names.
"""

from __future__ import annotations

from enum import Enum

#: The one component every model has. Never optional, needs no modality hooks,
#: so it is kept out of ``MODALITY_COMPONENTS``.
BACKBONE = "backbone"


class ComponentKind(str, Enum):
    """Where a component sits relative to the decoder.

    ``input_encoder`` components (vision, audio) run upstream and have their
    embeddings scattered into the token sequence; ``output_head`` components run
    downstream and change the generator's output instead. The distinction
    decides whether a component participates in fusion.
    """

    input_encoder = "input_encoder"
    output_head = "output_head"


#: Modality components in canonical order. Iteration order fixes the order of
#: fusion, per-component quantization, and export, so it is part of the contract.
MODALITY_COMPONENTS: dict[str, ComponentKind] = {
    "visual": ComponentKind.input_encoder,
    "audio": ComponentKind.input_encoder,
}

#: Every component the framework knows, backbone first.
ALL_COMPONENTS: tuple[str, ...] = (BACKBONE, *MODALITY_COMPONENTS)


def require_component(name: str) -> str:
    """Return ``name`` if it is a known modality component, else raise.

    The single source of the "unknown component" error, so every layer --
    schema, precision, cache, generator, sim collection -- rejects a bad
    component name identically.
    """
    if name not in MODALITY_COMPONENTS:
        raise KeyError(
            f"Unknown component {name!r}. "
            f"Known components: {sorted(MODALITY_COMPONENTS)}."
        )
    return name


def is_input_encoder(name: str) -> bool:
    return MODALITY_COMPONENTS.get(name) is ComponentKind.input_encoder
