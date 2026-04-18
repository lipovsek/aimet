# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DiskBackedModelCache."""

import pytest

from GenAILab.shared.helpers.model_cache import DiskBackedModelCache


class TestBuildKey:
    def test_deterministic(self):
        params = {"model_id": "org/model", "seq": 32, "ctx": 64}
        k1 = DiskBackedModelCache.build_key(params)
        k2 = DiskBackedModelCache.build_key(params)
        assert k1 == k2

    def test_order_independent(self):
        k1 = DiskBackedModelCache.build_key({"a": 1, "b": 2})
        k2 = DiskBackedModelCache.build_key({"b": 2, "a": 1})
        assert k1 == k2

    def test_different_params_different_keys(self):
        k1 = DiskBackedModelCache.build_key({"model_id": "a"})
        k2 = DiskBackedModelCache.build_key({"model_id": "b"})
        assert k1 != k2

    def test_key_is_hex_string(self):
        key = DiskBackedModelCache.build_key({"x": 1})
        assert len(key) == 16
        int(key, 16)  # Should not raise


class TestDiskBackedModelCache:
    def test_cache_miss(self, tmp_path):
        cache = DiskBackedModelCache(tmp_path / "model_cache")
        assert cache.get("nonexistent_key") is None

    def test_clear(self, tmp_path):
        cache_dir = tmp_path / "model_cache"
        cache = DiskBackedModelCache(cache_dir)
        # Just verify clear doesn't crash on empty cache
        cache.clear()
        assert cache._index["entries"] == {}

    def test_index_persists(self, tmp_path):
        cache_dir = tmp_path / "model_cache"
        cache1 = DiskBackedModelCache(cache_dir)
        # Manually add an entry to test index persistence
        cache1._index["entries"]["test_key"] = {"metadata": {}, "created": "now"}
        cache1._save_index()

        cache2 = DiskBackedModelCache(cache_dir)
        assert "test_key" in cache2._index["entries"]
