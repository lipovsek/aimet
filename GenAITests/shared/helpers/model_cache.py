# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Disk-backed cache for ONNX model exports that persists across pytest sessions."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import onnx
import torch
from transformers import AutoConfig, PretrainedConfig


@dataclass
class ModelCacheEntry:
    """Container for the ONNX model components returned by an export."""

    backbone: onnx.ModelProto
    visual: Optional[onnx.ModelProto] = None
    embedding: Optional[torch.nn.Embedding] = None
    config: Optional[PretrainedConfig] = None


class DiskBackedModelCache:
    """Disk-backed cache for exported ONNX models.

    Follows the same pattern as :class:`DiskBackedFPCache` but stores ONNX
    model protos (with external data) instead of tensors.

    The cache stores an ``index.json`` manifest alongside the model directories
    so that cached entries can be inspected without loading model data.

    Storage layout::

        {cache_dir}/
        +-- index.json
        +-- {key_hash}/
            +-- config.json
            +-- backbone/
            |   +-- model.onnx
            |   +-- model.data
            +-- visual/          (VLM only)
            |   +-- model.onnx
            |   +-- model.data
            +-- embedding.pth    (VLM only)
    """

    _INDEX_FILE = "index.json"
    _INDEX_VERSION = 1

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> ModelCacheEntry | None:
        """Return a cached :class:`ModelCacheEntry` if it exists on disk, else ``None``."""
        if key not in self._index["entries"]:
            return None

        entry_meta = self._index["entries"][key]
        entry_dir = self._cache_dir / key

        backbone_path = entry_dir / "backbone" / "model.onnx"
        if not backbone_path.exists():
            # Stale index entry -- files were deleted externally
            del self._index["entries"][key]
            self._save_index()
            return None

        # Staleness check: compare stored config against live HuggingFace config
        config_path = entry_dir / "config.json"
        if config_path.exists() and "model_id" in entry_meta.get("metadata", {}):
            try:
                stored_config = AutoConfig.from_pretrained(str(config_path))
                live_config = AutoConfig.from_pretrained(
                    entry_meta["metadata"]["model_id"]
                )
                if not _equivalent_configs(stored_config, live_config):
                    print(
                        f"Model cache: config changed for {entry_meta['metadata']['model_id']}, "
                        "invalidating cache entry."
                    )
                    shutil.rmtree(entry_dir, ignore_errors=True)
                    del self._index["entries"][key]
                    self._save_index()
                    return None
            except Exception:
                # If we can't check staleness (e.g. offline), trust the cache
                pass

        # Load components from disk
        backbone = onnx.load(str(backbone_path))

        visual_path = entry_dir / "visual" / "model.onnx"
        visual = onnx.load(str(visual_path)) if visual_path.exists() else None

        embedding_path = entry_dir / "embedding.pth"
        if embedding_path.exists():
            weights = torch.load(str(embedding_path), map_location="cpu")
            if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
                raise ValueError("Expected a 2D embedding tensor in embedding.pth")
            embedding = torch.nn.Embedding.from_pretrained(weights, freeze=False)
            embedding = embedding.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            embedding = None

        config = (
            AutoConfig.from_pretrained(str(config_path))
            if config_path.exists()
            else None
        )

        return ModelCacheEntry(
            backbone=backbone, visual=visual, embedding=embedding, config=config
        )

    def put(self, key: str, entry: ModelCacheEntry, metadata: Optional[dict] = None):
        """Write a :class:`ModelCacheEntry` to disk and update the index."""
        entry_dir = self._cache_dir / key
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Save backbone ONNX model with external data
        backbone_dir = entry_dir / "backbone"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        backbone_path = backbone_dir / "model.onnx"
        onnx.save_model(
            entry.backbone,
            str(backbone_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )

        # Save visual ONNX model if present
        if entry.visual is not None:
            visual_dir = entry_dir / "visual"
            visual_dir.mkdir(parents=True, exist_ok=True)
            visual_path = visual_dir / "model.onnx"
            onnx.save_model(
                entry.visual,
                str(visual_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model.data",
            )

        # Save embedding if present
        if entry.embedding is not None:
            embedding_path = entry_dir / "embedding.pth"
            torch.save(entry.embedding.state_dict()["weight"], str(embedding_path))

        # Save config if present
        if entry.config is not None:
            entry.config.save_pretrained(str(entry_dir))

        self._index["entries"][key] = {
            "metadata": metadata or {},
            "created": datetime.now().isoformat(),
        }
        self._save_index()

    def get_or_export(
        self,
        key: str,
        compute_fn: Callable[[], ModelCacheEntry],
        metadata: Optional[dict] = None,
    ) -> ModelCacheEntry:
        """Return cached entry for *key*, exporting and storing it if absent."""
        result = self.get(key)
        if result is not None:
            print(f"Model cache: hit for key {key}")
            return result
        print(f"Model cache: miss for key {key}, exporting...")
        result = compute_fn()
        self.put(key, result, metadata=metadata)
        return self.get(key)

    def clear(self):
        """Remove **all** cached data and reset the index."""
        shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._empty_index()

    @staticmethod
    def build_key(params: dict) -> str:
        """Return a deterministic hex digest of *params* for use as a cache key."""
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        index_path = self._cache_dir / self._INDEX_FILE
        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)
        return self._empty_index()

    def _save_index(self):
        index_path = self._cache_dir / self._INDEX_FILE
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @classmethod
    def _empty_index(cls) -> dict:
        return {"version": cls._INDEX_VERSION, "entries": {}}


def _equivalent_configs(config_a, config_b) -> bool:
    """Compare two HuggingFace configs by dict equality, ignoring ``_name_or_path``."""
    config_dict_a = config_a.to_dict()
    config_dict_b = config_b.to_dict()
    config_dict_a.pop("_name_or_path", None)
    config_dict_b.pop("_name_or_path", None)
    return config_dict_a == config_dict_b
