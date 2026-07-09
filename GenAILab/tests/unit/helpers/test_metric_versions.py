# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Pins each metric's scoring contract to its declared SCORING_VERSION.

If this fails, you changed scoring behavior without bumping SCORING_VERSION --
update it (and the fingerprint below) to match.
"""

import torch

from GenAILab.bench.datasets import LazyMMMUDataset
from GenAILab.bench.metrics import EvaluationMetric, MMMU


class _FakeTokenizer:
    """Minimal tokenizer stand-in with a controllable vocab for letter tokens."""

    def __init__(self, single_token_letters=("A", "B", "C", "D")):
        self._single_token_letters = set(single_token_letters)

    def __call__(self, text, add_special_tokens=False):
        stripped = text.strip()
        if text.startswith(" ") and stripped in self._single_token_letters:
            return {"input_ids": [hash(("space", stripped)) % 100000]}
        if stripped in self._single_token_letters:
            return {"input_ids": [hash(("bare", stripped)) % 100000]}
        return {"input_ids": [1, 2]}  # multi-token: forces bare-letter fallback


class TestEvaluationMetricDefaultVersion:
    def test_base_class_default_is_one(self):
        assert EvaluationMetric.SCORING_VERSION == 1


class TestMMMUScoringContractV2:
    def test_scoring_version_is_two(self):
        assert MMMU.SCORING_VERSION == 2

    def test_prefers_space_prefixed_letter_token(self):
        tokenizer = _FakeTokenizer(single_token_letters=("A", "B", "C", "D"))
        space_id = tokenizer(" A")["input_ids"][0]
        bare_id = tokenizer("A")["input_ids"][0]
        assert space_id != bare_id
        assert MMMU._token_id(tokenizer, "A") == space_id

    def test_falls_back_to_bare_letter_when_space_prefixed_is_multi_token(self):
        class _FallbackTokenizer:
            def __call__(self, text, add_special_tokens=False):
                if not text.startswith(" ") and text.strip() == "A":
                    return {"input_ids": [42]}
                return {"input_ids": [1, 2]}

        assert MMMU._token_id(_FallbackTokenizer(), "A") == 42

    def test_all_four_letters_resolve_to_distinct_ids(self):
        tokenizer = _FakeTokenizer(single_token_letters=("A", "B", "C", "D"))
        ids = [MMMU._token_id(tokenizer, letter) for letter in "ABCD"]
        assert len(set(ids)) == 4


class _FakeProcessor:
    """Records the messages passed to apply_chat_template for assertion."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.last_messages = None
        self.last_continue_final_message = None

    def apply_chat_template(self, messages, tokenize, continue_final_message, **kwargs):
        self.last_messages = messages
        self.last_continue_final_message = continue_final_message
        return "<rendered prompt>"

    def __call__(self, text, images, return_tensors):
        return {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class TestMMMUPromptContractV2:
    def _get_sample(self):
        raw_dataset = [
            {
                "question": "What is 2 + 2?",
                "options": ["3", "4", "5", "6"],
                "answer": "B",
            }
        ]
        processor = _FakeProcessor()
        dataset = LazyMMMUDataset(raw_dataset, processor, context_length=128)
        dataset[0]
        return processor

    def test_answer_prompt_is_in_an_assistant_turn(self):
        processor = self._get_sample()
        messages = processor.last_messages
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Answer:"
        assert messages[0]["role"] == "user"

    def test_continues_the_final_message(self):
        processor = self._get_sample()
        assert processor.last_continue_final_message is True
