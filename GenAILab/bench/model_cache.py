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
    extras: Optional[dict[str, torch.nn.Module]] = None


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
            print(
                f"  [cache] Model cache stale entry for key {key} "
                f"(backbone not found on disk) — removing"
            )
            del self._index["entries"][key]
            self._save_index()
            return None

        # Staleness check: compare hub config hash at cache time vs now.
        # We can't do this for small models, since the live config would be different.
        small_model = entry_meta.get("metadata", {}).get("small_model", False)
        stored_hash = entry_meta.get("hub_config_hash")
        model_id = entry_meta.get("metadata", {}).get("model_id")
        if stored_hash is not None and model_id is not None and not small_model:
            try:
                current_hash = _hub_config_hash(model_id)
                if stored_hash != current_hash:
                    print(
                        f"  [cache] Model cache config changed for {model_id} — invalidating"
                    )
                    shutil.rmtree(entry_dir, ignore_errors=True)
                    del self._index["entries"][key]
                    self._save_index()
                    return None
            except Exception:
                # If we can't check staleness (e.g. offline), trust the cache
                pass

        # Load components from disk
        config_path = entry_dir / "config.json"
        backbone = onnx.load(str(backbone_path))

        visual_path = entry_dir / "visual" / "model.onnx"
        visual = onnx.load(str(visual_path)) if visual_path.exists() else None

        embedding_path = entry_dir / "embedding.pth"
        if embedding_path.exists():
            weights = torch.load(
                str(embedding_path), map_location="cpu", weights_only=True
            )
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

        # Load extras: any .pth files not already handled (embedding.pth)
        extras_dir = entry_dir / "extras"
        extras = {}
        if extras_dir.exists():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for pth_file in sorted(extras_dir.glob("*.pth")):
                tensor = torch.load(
                    str(pth_file), map_location="cpu", weights_only=True
                )
                name = pth_file.stem
                module = torch.nn.Embedding(*tensor.shape)
                module.weight = torch.nn.Parameter(tensor, requires_grad=False)
                extras[name] = module.to(device)

        return ModelCacheEntry(
            backbone=backbone,
            visual=visual,
            embedding=embedding,
            config=config,
            extras=extras or None,
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

        # Save extras (model-specific auxiliary modules)
        if entry.extras:
            extras_dir = entry_dir / "extras"
            extras_dir.mkdir(parents=True, exist_ok=True)
            for name, module in entry.extras.items():
                if module is not None:
                    torch.save(
                        module.state_dict()["weight"],
                        str(extras_dir / f"{name}.pth"),
                    )

        # Save config if present
        if entry.config is not None:
            entry.config.save_pretrained(str(entry_dir))

        # Compute hub config hash for staleness checks.
        # entry.config may have been mutated at runtime (attn_implementation,
        # torch_dtype, etc.), so we fetch the pristine hub config separately.
        hub_config_hash = None
        model_id = (metadata or {}).get("model_id")
        if model_id is not None:
            try:
                hub_config_hash = _hub_config_hash(model_id)
            except Exception:
                pass

        index_entry = {
            "metadata": metadata or {},
            "created": datetime.now().isoformat(),
        }
        if hub_config_hash is not None:
            index_entry["hub_config_hash"] = hub_config_hash

        self._index["entries"][key] = index_entry
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
            _log_model_cache_hit(key, metadata)
            return result
        _log_model_cache_miss(key, metadata)
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
            try:
                with open(index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"  [cache] Model cache index corrupted — resetting")
                index_path.unlink()
        return self._empty_index()

    def _save_index(self):
        index_path = self._cache_dir / self._INDEX_FILE
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @classmethod
    def _empty_index(cls) -> dict:
        return {"version": cls._INDEX_VERSION, "entries": {}}


_CACHE_BANNER = "\n" + "=" * 70 + "\n"


def _model_label(metadata: dict | None) -> str:
    if metadata:
        model_id = metadata.get("model_id", "")
        if model_id:
            return model_id
    return ""


def _log_model_cache_hit(key: str, metadata: dict | None):
    label = _model_label(metadata)
    print(_CACHE_BANNER, end="")
    print(f"  MODEL CACHE HIT")
    if label:
        print(f"  Model: {label}")
    print(f"  Loading exported ONNX model from disk")
    print("=" * 70 + "\n")


def _log_model_cache_miss(key: str, metadata: dict | None):
    label = _model_label(metadata)
    detail = f" for {label}" if label else ""
    print(f"  [cache] Model cache miss{detail} — exporting from scratch")


def _strip_private_keys(obj):
    """Recursively remove ``_``-prefixed keys from nested dicts.

    VLM configs contain nested sub-configs (``text_config``, ``vision_config``,
    etc.) that each carry their own ``_name_or_path`` which can resolve to
    different HuggingFace cache snapshot paths between process invocations.
    A top-level-only filter misses these, producing non-deterministic hashes.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_private_keys(v) for k, v in obj.items() if not k.startswith("_")
        }
    if isinstance(obj, list):
        return [_strip_private_keys(item) for item in obj]
    return obj


def _hub_config_hash(model_id: str) -> str:
    """Return a deterministic hash of the pristine hub config for *model_id*.

    Fetches the config directly from HuggingFace (or local cache) and hashes
    its canonical JSON representation.  Private keys (``_``-prefixed) are
    recursively excluded so that local runtime metadata (like snapshot paths
    in nested sub-configs) does not affect the hash.
    """
    config = AutoConfig.from_pretrained(model_id)
    config_dict = _strip_private_keys(config.to_dict())
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
