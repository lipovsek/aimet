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


def thinking_kwargs(chat_template) -> dict:
    """``{"enable_thinking": False}`` iff the template actually branches on it.

    ``apply_chat_template`` accepts arbitrary ``**kwargs`` and forwards
    whichever ones the template's Jinja source references as free variables;
    anything else is dead weight. For a template that never mentions
    ``enable_thinking`` (most non-reasoning chat templates), passing it
    unconditionally doesn't change the render, but it does trip
    ``transformers``' own "kwargs passed to `processor.__call__` have to be in
    `processor_kwargs`" warning on every single call -- the kwarg's introspection
    (``jinja2.meta.find_undeclared_variables``) doesn't find it declared,
    reclassifies it as a processor kwarg, and logs. Harmless (nothing downstream
    consumes it when ``tokenize=False``, which is how this codebase always calls
    it), but it fires once per sample.

    ``chat_template`` may be a plain template string or transformers' newer
    ``{name: template}`` dict form (multiple named templates); ``None`` means no
    template is available at all. A plain substring check is used rather than
    parsing the Jinja AST (as transformers does internally) -- cheaper, and
    sufficient: a template that references ``enable_thinking`` as a real
    variable necessarily spells the name somewhere in its source.
    """
    if isinstance(chat_template, dict):
        chat_template = chat_template.get(
            "default", next(iter(chat_template.values()), None)
        )
    if chat_template and "enable_thinking" in chat_template:
        return {"enable_thinking": False}
    return {}
