# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared text-generation helpers used by metrics and datasets."""

import torch
from transformers import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria

from GenAILab.qai_hub_lm.models.generator import Generator


class ContextLengthStoppingCriteria(StoppingCriteria):
    """Stop generation when reaching the model's context length limit.

    This stopping criterion monitors the total sequence length (input + generated tokens)
    and stops generation when reaching the Generator's context_length, preventing
    attempts to exceed the model's maximum supported context.

    In autoregressive generation, each decode step adds exactly 1 token, so we can
    safely generate up to the exact context_length without wasting any capacity.

    Args:
        context_length: Maximum context length supported by the model
        verbose: If True, prints a message when stopping (default: False)
    """

    def __init__(
        self, context_length: int, sequence_lengths: list[int], verbose: bool = False
    ):
        self.context_length = context_length
        self.min_sequence_length = min(sequence_lengths)
        self.verbose = verbose
        self._warned = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """Check if generation should stop based on sequence length.

        Args:
            input_ids: Current sequence of token ids (batch_size, seq_len)
            scores: Model output scores (not used, required by interface)
            **kwargs: Additional generation state (may include past_key_values)

        Returns:
            True if generation should stop (at or exceeding context_length), False otherwise
        """
        current_len = input_ids.shape[-1]
        next_len = current_len + self.min_sequence_length
        should_stop = next_len >= self.context_length

        if should_stop and self.verbose and not self._warned:
            print(
                f"ContextLengthStoppingCriteria: Reached context length limit "
                f"({current_len}/{self.context_length} tokens)"
            )
            self._warned = True

        return should_stop


def build_generation_config(model, tokenizer, **overrides) -> GenerationConfig:
    """Build a GenerationConfig, merging EOS tokens from model config and tokenizer.

    Args:
        model: Generator model instance (may have .context_length attribute)
        tokenizer: Tokenizer instance with eos_token_id and pad_token_id
        **overrides: Additional GenerationConfig parameters to override defaults

    Returns:
        GenerationConfig with sensible defaults and merged EOS tokens
    """
    eos_ids = set()
    for src in (
        getattr(model.config, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
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
        max_new_tokens=2048,
    )
    defaults.update(overrides)
    return GenerationConfig(**defaults)
