# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared text-generation helpers used by metrics and datasets."""

from transformers import GenerationConfig


def build_generation_config(model, tokenizer, **overrides) -> GenerationConfig:
    """Build a GenerationConfig, merging EOS tokens from model config and tokenizer."""
    eos_ids = set()
    for src in (
        getattr(model.config, "eos_token_id", None),
        tokenizer.eos_token_id,
    ):
        if src is None:
            continue
        if isinstance(src, (list, tuple)):
            eos_ids.update(src)
        else:
            eos_ids.add(src)

    defaults = dict(
        eos_token_id=sorted(eos_ids) if eos_ids else tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8,
    )
    defaults.update(overrides)
    return GenerationConfig(**defaults)
