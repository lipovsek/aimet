# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dataset utilities."""

import torch

from GenAILab.bench.datasets import ChunkedDataset, MMLU, MMMLU


class TestChunkedDataset:
    def test_len(self):
        data = {
            "input_ids": torch.randint(0, 100, (1, 200)),
            "attention_mask": torch.ones(1, 200, dtype=torch.int),
        }
        ds = ChunkedDataset(data, size_per_chunk=50)
        assert len(ds) == 4  # 200 // 50

    def test_getitem(self):
        data = {
            "input_ids": torch.arange(100).unsqueeze(0),
            "attention_mask": torch.ones(1, 100, dtype=torch.int),
        }
        ds = ChunkedDataset(data, size_per_chunk=25)
        item = ds[1]  # second chunk: indices 25-49
        assert item["input_ids"].shape == (1, 25)
        assert item["input_ids"][0, 0].item() == 25
        assert item["input_ids"][0, -1].item() == 49

    def test_shapes(self):
        context_length = 64
        data = {
            "input_ids": torch.randint(0, 100, (1, 256)),
            "attention_mask": torch.ones(1, 256, dtype=torch.int),
        }
        ds = ChunkedDataset(data, size_per_chunk=context_length)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].shape == (1, context_length)
            assert item["attention_mask"].shape == (1, context_length)


class TestMMLUFormatting:
    def test_format_question(self):
        result = MMLU._format_question(
            "What is 2+2?",
            ["3", "4", "5", "6"],
        )
        assert "A. 3" in result
        assert "B. 4" in result
        assert "C. 5" in result
        assert "D. 6" in result
        assert result.endswith("Answer:")

    def test_format_question_and_answer(self):
        result = MMLU._format_question_and_answer(
            "What is 2+2?",
            ["3", "4", "5", "6"],
            "B",
        )
        assert result.endswith("Answer: B")


class TestMMMluFormatting:
    def test_format_question(self):
        result = MMMLU._format_question(
            "What is 2+2?",
            ("3", "4", "5", "6"),
        )
        assert "A. 3" in result
        assert result.endswith("Answer:")
