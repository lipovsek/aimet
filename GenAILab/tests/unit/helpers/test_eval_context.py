# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for EvaluationContext."""

import pytest
import torch

from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.bench.eval_context import EvaluationContext


@pytest.fixture
def fp_cache(tmp_path):
    return DiskBackedFPCache(tmp_path / "fp_cache")


@pytest.fixture
def eval_ctx(fp_cache):
    model_args = {"model_id": "test-model", "sequence_length": 32, "context_length": 64}
    return EvaluationContext(fp_cache=fp_cache, model_args=model_args)


class TestEvaluationContext:
    def test_get_or_compute_fp_caches(self, eval_ctx):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"logits": torch.randn(10, 4)}

        result1 = eval_ctx.get_or_compute_fp("mmlu_logits", compute)
        result2 = eval_ctx.get_or_compute_fp("mmlu_logits", compute)
        assert call_count == 1
        assert torch.equal(result1["logits"], result2["logits"])

    def test_get_or_compute_quant_caches(self, eval_ctx):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"logits": torch.randn(10, 4)}

        result1 = eval_ctx.get_or_compute_quant("mmlu_logits", compute)
        result2 = eval_ctx.get_or_compute_quant("mmlu_logits", compute)
        assert call_count == 1
        assert result1 is result2

    def test_quant_cache_isolated(self, fp_cache):
        args = {"model_id": "test", "seq": 32}
        ctx1 = EvaluationContext(fp_cache=fp_cache, model_args=args)
        ctx2 = EvaluationContext(fp_cache=fp_cache, model_args=args)

        ctx1.get_or_compute_quant("mmlu", lambda: {"x": torch.tensor(1.0)})
        # ctx2 should not have ctx1's quant cache
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"x": torch.tensor(2.0)}

        result = ctx2.get_or_compute_quant("mmlu", compute)
        assert call_count == 1
        assert result["x"].item() == 2.0

    def test_fp_cache_shared(self, fp_cache):
        args = {"model_id": "test", "seq": 32}
        ctx1 = EvaluationContext(fp_cache=fp_cache, model_args=args)
        ctx2 = EvaluationContext(fp_cache=fp_cache, model_args=args)

        data = {"logits": torch.randn(5, 4)}
        ctx1.get_or_compute_fp("mmlu", lambda: data)

        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"logits": torch.zeros(5, 4)}

        result = ctx2.get_or_compute_fp("mmlu", compute)
        assert call_count == 0
        assert torch.equal(result["logits"], data["logits"])

    def test_hash_dict_deterministic(self):
        d = {"model_id": "test", "seq": 32, "ctx": 64}
        h1 = EvaluationContext._hash_dict(d)
        h2 = EvaluationContext._hash_dict(d)
        assert h1 == h2

    def test_hash_dict_order_independent(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert EvaluationContext._hash_dict(d1) == EvaluationContext._hash_dict(d2)
