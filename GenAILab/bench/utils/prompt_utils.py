# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for loading text prompts used by metrics and datasets."""

from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TEXT_PROMPTS_FILE = PROMPTS_DIR / "text_prompts.yaml"
CALIBRATION_PROMPTS_FILE = PROMPTS_DIR / "calibration_prompts.yaml"


def normalize_prompt(entry) -> str:
    """Coerce a raw prompt YAML entry into a plain string."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        k, v = next(iter(entry.items()))
        return f"{k}: {v}"
    return str(entry)


def load_text_prompts(
    num_prompts: int | None = None,
    prompts_file: Path | str = TEXT_PROMPTS_FILE,
) -> list[str]:
    """Load (and normalize) prompts from ``prompts_file``.

    Defaults to the evaluation prompt set (:data:`TEXT_PROMPTS_FILE`); pass
    :data:`CALIBRATION_PROMPTS_FILE` for self-generated calibration so the two
    sets stay disjoint. Returns all prompts when ``num_prompts`` is ``None``,
    otherwise the first ``num_prompts`` entries in file order.
    """
    with open(prompts_file) as f:
        raw_prompts = yaml.safe_load(f)
    prompts = [normalize_prompt(p) for p in raw_prompts]
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts
