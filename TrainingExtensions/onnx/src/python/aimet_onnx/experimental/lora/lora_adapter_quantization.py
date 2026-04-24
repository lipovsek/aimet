# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LoRA quantization utilities for integrating with QuantizationSimModel.

Standalone functions that interact with QuantSim's qc_quantize_op_dict to
configure, freeze, and unfreeze quantizers for multi-adapter calibration
workflows. All functions take ``lora_names: dict`` with ``"params"``,
``"activations"``, and ``"scales"`` keys, as returned by
``configure_lora_onnx()`` or ``prepare_lora_onnx()``.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import onnx
from aimet_onnx.common.defs import qtype

logger = logging.getLogger(__name__)


def _natural_sort_key(s: str) -> list:
    """Sort key so ``layers.2`` sorts before ``layers.10``."""
    return [
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", s)
    ]


# ---------------------------------------------------------------------------
# Quantizer configuration
# ---------------------------------------------------------------------------


def set_lora_bitwidth(
    sim,
    lora_names: dict[str, list[str]],
    param_type: str | qtype,
    activation_type: str | qtype,
) -> int:
    """Set bitwidth for LoRA parameter and activation quantizers.

    :param sim: QuantizationSimModel instance
    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :param param_type: Quantization type for LoRA weights (e.g., ``int16``).
    :param activation_type: Quantization type for LoRA activations (e.g., ``int8``).
    :return: Number of quantizers updated
    :raises RuntimeError: If any LoRA quantizer is already frozen
    """
    if isinstance(param_type, str):
        param_type = qtype.from_string(param_type)
    if isinstance(activation_type, str):
        activation_type = qtype.from_string(activation_type)
    _, param_bw = param_type.to_legacy_repr()
    _, activation_bw = activation_type.to_legacy_repr()

    count = 0
    for name, bw in _iter_lora_quantizers(sim, lora_names, param_bw, activation_bw):
        qtzr = sim.qc_quantize_op_dict[name]
        if qtzr.is_encoding_frozen():
            raise RuntimeError(
                f"LoRA quantizer '{name}' is frozen. "
                f"Call set_lora_bitwidth() BEFORE freeze functions."
            )
        qtzr.set_bitwidth(bw)
        count += 1

    if count == 0:
        all_names = lora_names["params"] + lora_names["activations"]
        raise ValueError(
            f"No LoRA quantizers found in sim for {len(all_names)} LoRA names. "
            f"First few names: {all_names[:3]}"
        )

    # Disable scale quantizers — scales are scalar constants (alpha/rank),
    # not statistical distributions.  QuantSim wraps them as activation
    # quantizers because Mul is not in OPS_WITH_PARAMS, but quantizing a
    # single constant adds no value and breaks the freeze/unfreeze lifecycle.
    scale_count = 0
    for name in lora_names.get("scales", {}):
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = False
            scale_count += 1

    logger.info(
        "Set LoRA bitwidth: %d quantizers (params=%d-bit, activations=%d-bit), "
        "disabled %d scale quantizers",
        count,
        param_bw,
        activation_bw,
        scale_count,
    )
    return count


def _iter_lora_quantizers(sim, lora_names, param_bw, activation_bw):
    """Yield (name, bitwidth) for each LoRA quantizer found in sim."""
    for name in lora_names["params"]:
        if name in sim.qc_quantize_op_dict:
            yield name, param_bw
    for name in lora_names["activations"]:
        if name in sim.qc_quantize_op_dict:
            yield name, activation_bw


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------


def freeze_base_param_quantizers(sim, lora_names: dict[str, list[str]]) -> int:
    """Freeze all base (non-LoRA) parameter quantizers.

    :param sim: QuantizationSimModel instance
    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :return: Number of quantizers frozen
    """
    lora_set = set(lora_names["params"])
    count = 0
    for name in sim.param_names:
        if name not in lora_set and name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            if qtzr.enabled:
                qtzr.freeze_encodings()
                count += 1

    logger.info("Froze %d base param quantizers", count)
    return count


def freeze_base_activation_quantizers(sim, lora_names: dict[str, list[str]]) -> int:
    """Freeze all base (non-LoRA) activation quantizers.

    :param sim: QuantizationSimModel instance
    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :return: Number of quantizers frozen
    """
    lora_set = set(lora_names["activations"])
    count = 0
    for name in sim.activation_names:
        if name not in lora_set and name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            if qtzr.enabled:
                qtzr.freeze_encodings()
                count += 1

    logger.info("Froze %d base activation quantizers", count)
    return count


def freeze_base_model(sim, lora_names: dict[str, list[str]]) -> int:
    """Freeze all base quantizers (params + activations).

    :param sim: QuantizationSimModel instance
    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :return: Total number of quantizers frozen
    """
    count = freeze_base_param_quantizers(sim, lora_names)
    count += freeze_base_activation_quantizers(sim, lora_names)
    logger.info("Froze %d total base quantizers", count)
    return count


def unfreeze_lora_quantizers(sim, lora_names: dict[str, list[str]]) -> int:
    """Unfreeze all LoRA quantizers (params + activations) and reset stats.

    :param sim: QuantizationSimModel instance
    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :return: Number of quantizers unfrozen
    """
    all_lora = lora_names["params"] + lora_names["activations"]
    count = 0
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            qtzr._is_encoding_frozen = False  # pylint: disable=protected-access
            qtzr.reset_encoding_stats()
            count += 1

    if count == 0:
        raise ValueError(
            f"No LoRA quantizers found in sim for {len(all_lora)} LoRA names. "
            f"First few names: {all_lora[:3]}"
        )

    logger.info("Unfroze %d LoRA quantizers", count)
    return count


# ---------------------------------------------------------------------------
# Encoding snapshot / restore
# ---------------------------------------------------------------------------


def get_lora_encodings(
    sim,
    lora_names: dict[str, list[str]],
    encoding_version: str = "1.0.0",
) -> dict[str, dict]:
    """Capture current LoRA quantizer encodings (params + activations).

    :param sim: QuantizationSimModel instance
    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :param encoding_version: Encoding format version (default ``"1.0.0"``)
    :return: Dict mapping LoRA quantizer name to encoding dict
    :raises RuntimeError: If any LoRA quantizer has no encoding (uncalibrated)
    """
    all_lora = lora_names["params"] + lora_names["activations"]
    encodings = {}
    for name in all_lora:
        if name not in sim.qc_quantize_op_dict:
            continue
        qtzr = sim.qc_quantize_op_dict[name]
        encoding = qtzr.export_encodings(encoding_version)
        if encoding is None:
            raise RuntimeError(
                f"LoRA quantizer '{name}' has no encoding. "
                f"Call sim.compute_encodings() before get_lora_encodings()."
            )
        encodings[name] = encoding

    if not encodings:
        all_lora = lora_names["params"] + lora_names["activations"]
        raise ValueError(
            f"No LoRA quantizers found in sim for {len(all_lora)} LoRA names. "
            f"First few names: {all_lora[:3]}"
        )

    logger.info("Captured encodings for %d LoRA quantizers", len(encodings))
    return encodings


def set_lora_encodings(sim, encodings: dict[str, dict]) -> int:
    """Restore previously captured LoRA quantizer encodings and freeze them.

    :param sim: QuantizationSimModel instance
    :param encodings: Dict from ``get_lora_encodings()``
    :return: Number of quantizers updated
    :raises ValueError: If any encoding name is not found in the sim
    """
    missing = [name for name in encodings if name not in sim.qc_quantize_op_dict]
    if missing:
        raise ValueError(
            f"{len(missing)} encoding names not found in sim: {missing[:5]}"
        )

    count = 0
    for name, encoding_dict in encodings.items():
        qtzr = sim.qc_quantize_op_dict[name]
        qtzr._is_encoding_frozen = False  # pylint: disable=protected-access
        qtzr._load_encodings_dict(encoding_dict)  # pylint: disable=protected-access
        qtzr.freeze_encodings()
        count += 1

    logger.info("Restored and froze encodings for %d LoRA quantizers", count)
    return count


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def get_zero_weights(
    model: onnx.ModelProto, lora_names: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """Return zero-valued weights for all LoRA params.

    :param model: ONNX ModelProto with LoRA initializers
    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :return: Dict mapping LoRA param name to zero numpy array
    :raises ValueError: If any param name is not found in model initializers
    """
    init_map = {init.name: init for init in model.graph.initializer}
    param_names = lora_names["params"]

    missing = [name for name in param_names if name not in init_map]
    if missing:
        raise ValueError(
            f"{len(missing)} LoRA names not found in model initializers: {missing[:5]}. "
            f"Ensure the model was prepared with configure_lora_onnx()."
        )

    zero_weights = {}
    for name in param_names:
        init = init_map[name]
        shape = tuple(init.dims)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        zero_weights[name] = np.zeros(shape, dtype=dtype)

    return zero_weights


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------


def get_adapter_scale_weights(
    lora_names: dict,
    adapter_path: str,
) -> dict[str, np.ndarray]:
    """Return LoRA scale values for a specific adapter as a feed dict.

    Reads ``adapter_config.json`` from ``adapter_path``, computes
    ``lora_alpha / r``, and returns a dict mapping each scale name in
    ``lora_names["scales"]`` to a ``np.float32`` scalar.

    For modules not in the adapter's ``target_modules``, the default
    scale value from ``lora_names["scales"]`` is used (the zero-weight
    branch produces zero regardless of scale).

    :param lora_names: Dict with ``"scales"`` key as returned by
        ``configure_lora_onnx()``.
    :param adapter_path: Path to adapter directory containing
        ``adapter_config.json``.
    :return: Dict mapping scale name to ``np.float32`` scalar, suitable
        for passing to ``session.run()``.
    :raises FileNotFoundError: If ``adapter_config.json`` is not found.
    """
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    alpha = config.get("lora_alpha", config.get("r", 1))
    r = config.get("r", 1)
    adapter_scale = float(alpha) / float(r)
    target_modules = set(config.get("target_modules", []))

    scales_dict = lora_names.get("scales", {})
    result = {}

    for scale_name, default_value in scales_dict.items():
        # Extract module name from scale name
        # e.g., "base_model.model.layers.0.self_attn.q_proj.lora_scale" → "q_proj"
        module_name = _extract_module_name_from_scale(scale_name)
        if module_name in target_modules:
            result[scale_name] = np.array(adapter_scale, dtype=np.float32)
        else:
            result[scale_name] = np.array(default_value, dtype=np.float32)

    return result


def _extract_module_name_from_scale(scale_name: str) -> str:
    """Extract the target module name from a scale initializer name.

    Handles both single-adapter and multi-adapter naming:
    - Single: ``"base_model.model...q_proj.lora_scale"`` → ``"q_proj"``
    - Multi:  ``"base_model.model...q_proj.lora_scale.code"`` → ``"q_proj"``
    """
    # Strip adapter name suffix if present (multi-adapter)
    # Pattern: ...module.lora_scale or ...module.lora_scale.{adapter}
    if ".lora_scale." in scale_name:
        # Multi-adapter: strip everything from .lora_scale onwards
        base = scale_name.split(".lora_scale.")[0]
    else:
        # Single-adapter: strip .lora_scale suffix
        base = scale_name.replace(".lora_scale", "")
    parts = base.split(".")
    return parts[-1] if parts else ""


# ---------------------------------------------------------------------------
# Multi-adapter helpers
# ---------------------------------------------------------------------------


def get_adapter_names(lora_names: dict) -> list[str]:
    """Return list of adapter names from lora_names.

    :param lora_names: Dict as returned by ``prepare_lora_onnx()``.
    :return: Sorted list of adapter names, or ``["default"]`` for single-adapter.
    """
    if "adapters" in lora_names:
        return sorted(lora_names["adapters"].keys())
    return ["default"]


def get_adapter_lora_names(lora_names: dict, adapter_name: str) -> dict:
    """Get lora_names subset for a specific adapter.

    :param lora_names: Dict as returned by ``prepare_lora_onnx()``.
    :param adapter_name: Name of the adapter.
    :return: Per-adapter dict with ``"params"``, ``"activations"``, ``"scales"``.
        For single-adapter, returns the full lora_names (backward-compat).
    """
    if "adapters" in lora_names and adapter_name in lora_names["adapters"]:
        return lora_names["adapters"][adapter_name]
    return lora_names


def build_concurrent_feed_dict(
    model: "onnx.ModelProto",
    lora_names: dict,
    active_adapters: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Build a complete feed dict for LoRA calibration or inference.

    For each adapter in the graph:
    - If in ``active_adapters``: uses provided weights (remapped to ONNX names)
    - If NOT in ``active_adapters``: uses zero weights

    Handles safetensors key remapping internally — user passes raw
    safetensors dicts (without adapter name in keys), and this function
    maps them to the multi-adapter ONNX init names.

    :param model: ONNX ModelProto with LoRA initializers.
    :param lora_names: Dict as returned by ``prepare_lora_onnx()``.
    :param active_adapters: Dict mapping adapter_name → weights dict
        (from ``safetensors.numpy.load_file()``). Adapters not in this dict
        are zeroed.
    :return: Feed dict mapping ONNX init names to numpy arrays.

    Example::

        code_wts = load_file("code/adapter_model.safetensors")
        feed = build_concurrent_feed_dict(
            model, lora_names,
            active_adapters={"code": code_wts}
        )
        sim.compute_encodings(lambda sess: calibrate(sess, feed))
    """
    init_map = {init.name: init for init in model.graph.initializer}
    graph_input_names = {inp.name for inp in model.graph.input}
    is_multi = "adapters" in lora_names
    adapter_names = get_adapter_names(lora_names)
    all_param_names = set(lora_names["params"])
    tp = lora_names.get("transposed_params")

    feed = {}

    for adapter_name in adapter_names:
        adapter_ln = get_adapter_lora_names(lora_names, adapter_name)
        adapter_params = adapter_ln["params"]

        if adapter_name in active_adapters:
            raw_weights = active_adapters[adapter_name]
            remapped = _remap_safetensors_to_onnx(
                raw_weights,
                adapter_name,
                is_multi,
                onnx_param_names=all_param_names,
                transposed_params=tp,
            )
            for param_name in adapter_params:
                # Only include if it's a graph input (promoted for calibration)
                if param_name not in graph_input_names:
                    continue
                if param_name in remapped:
                    feed[param_name] = remapped[param_name]
                elif param_name in init_map:
                    init = init_map[param_name]
                    shape = tuple(init.dims)
                    feed[param_name] = np.zeros(shape, dtype=np.float32)
        else:
            # Zero out this adapter
            for param_name in adapter_params:
                if param_name not in graph_input_names:
                    continue
                if param_name in init_map:
                    init = init_map[param_name]
                    shape = tuple(init.dims)
                    feed[param_name] = np.zeros(shape, dtype=np.float32)

    # Include scales only if they're in graph.input (promoted for calibration)
    scales_dict = lora_names.get("scales", {})
    for scale_name, default_value in scales_dict.items():
        if scale_name in graph_input_names:
            feed[scale_name] = np.array(default_value, dtype=np.float32)

    return feed


def _remap_safetensors_to_onnx(
    raw_weights: dict[str, np.ndarray],
    adapter_name: str,
    is_multi: bool,
    onnx_param_names: set[str] | None = None,
    transposed_params: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, np.ndarray]:
    """Remap PEFT safetensors keys to ONNX initializer names.

    Handles three mismatches:

    1. **Multi-adapter insertion**: PEFT saves ``lora_A.weight`` but
       multi-adapter ONNX uses ``lora_A.{adapter}.weight``.

    2. **Prefix mismatch**: HuggingFace CausalLM models have a ``.model``
       attribute wrapping the base model, so PEFT saves keys like
       ``base_model.model.model.layers...`` while ONNX node names produce
       ``base_model.model.layers...``. When ``onnx_param_names`` is provided,
       falls back to suffix matching if direct lookup fails.

    3. **Shape transposition**: ``dynamo=False`` stores weights in MatMul
       convention (transposed vs PEFT). When ``transposed_params`` is provided,
       transposes the weight data so it matches the ONNX graph's expected shape.

    :param raw_weights: Dict from ``safetensors.numpy.load_file()``.
    :param adapter_name: Name of the adapter.
    :param is_multi: Whether this is a multi-adapter graph.
    :param onnx_param_names: Set of ONNX LoRA param names for suffix matching.
        When provided, enables robust matching across prefix conventions.
    :param transposed_params: Dict mapping ONNX param name to its
        MatMul-convention shape. Weights matching these names are transposed
        from PEFT convention to MatMul convention before returning.
    :return: Dict with keys matching ONNX init names, data in graph convention.
    """
    # Step 1: Multi-adapter name insertion (lora_A.weight → lora_A.{adapter}.weight)
    if is_multi:
        step1 = {}
        for key, value in raw_weights.items():
            parts = key.split(".")
            new_parts = []
            for i, part in enumerate(parts):
                new_parts.append(part)
                if part in ("lora_A", "lora_B") and i + 1 < len(parts):
                    new_parts.append(adapter_name)
            step1[".".join(new_parts)] = value
    else:
        step1 = dict(raw_weights)

    # Step 2: If ONNX param names provided, match by suffix for any
    # keys that don't directly match (handles prefix mismatches)
    if onnx_param_names is None:
        return step1

    remapped = {}
    unmatched = {}
    for key, value in step1.items():
        if key in onnx_param_names:
            remapped[key] = value
        else:
            unmatched[key] = value

    if not unmatched:
        return remapped

    # Build suffix index from ONNX names for efficient lookup.
    # Suffix = everything from "layers." onward (covers all HF model hierarchies).
    suffix_to_onnx = {}
    for name in onnx_param_names:
        for marker in ("layers.", "lm_head.", "embed_tokens."):
            idx = name.find(marker)
            if idx >= 0:
                suffix_to_onnx[name[idx:]] = name
                break

    matched_count = 0
    for key, value in unmatched.items():
        for marker in ("layers.", "lm_head.", "embed_tokens."):
            idx = key.find(marker)
            if idx >= 0:
                suffix = key[idx:]
                if suffix in suffix_to_onnx:
                    remapped[suffix_to_onnx[suffix]] = value
                    matched_count += 1
                break

    if matched_count > 0:
        logger.info(
            "Suffix-matched %d/%d safetensors keys to ONNX names "
            "(prefix mismatch: safetensors vs ONNX naming convention)",
            matched_count,
            len(unmatched),
        )

    # Step 3: Transpose PEFT-convention weights to MatMul convention
    # for dynamo=False exports where the graph expects transposed shapes.
    if transposed_params:
        transposed_count = 0
        for name in list(remapped.keys()):
            if name in transposed_params:
                expected_shape = transposed_params[name]
                data = remapped[name]
                if data.ndim == 2 and data.shape != expected_shape:
                    remapped[name] = data.T
                    transposed_count += 1
        if transposed_count > 0:
            logger.info(
                "Transposed %d weights from PEFT to MatMul convention",
                transposed_count,
            )

    return remapped


# ---------------------------------------------------------------------------
# Encoding mapping (base model → PeftModel)
# ---------------------------------------------------------------------------


def adapt_base_encodings_for_lora(
    base_encodings: dict,
    target_modules: list[str] | set[str],
) -> dict:
    """Map base model encoding names to PeftModel encoding names.

    When using the two-phase workflow (base recipes on clean export, then LoRA
    on PeftModel export), the encoding names differ:

    - Base: ``/model/layers.0/self_attn/q_proj/Conv_output_0``
    - Peft: ``/base_model/model/layers.0/self_attn/q_proj/base_layer/Conv_output_0``

    This function transforms base encoding names so they can be loaded into a
    PeftModel QuantSim via ``sim.load_encodings(adapted, partial=True)``.

    Transforms applied:
    1. For target modules: insert ``/base_layer`` after ``/{target_module}``
    2. Add ``/base_model`` prefix (PeftModel wraps the model)

    Ported from the Torch notebook's ``adapt_base_encodings_for_peft_model()``
    in ``common/lora_utils/mpp_lora_utils.py``.

    :param base_encodings: Encodings dict from base model QuantSim export
        (has ``"activation_encodings"`` and ``"param_encodings"`` keys).
    :param target_modules: Set of target module names (e.g., ``["q_proj", "v_proj"]``).
    :return: New encodings dict with transformed names.
    """
    import copy

    adapted = copy.deepcopy(base_encodings)
    target_set = set(target_modules)

    for section in ("activation_encodings", "param_encodings"):
        if section not in adapted:
            continue
        encodings = adapted[section]

        if isinstance(encodings, list):
            # 1.0.0 format: list of dicts with "name" field
            for entry in encodings:
                if "name" in entry:
                    entry["name"] = _map_base_to_peft_name(entry["name"], target_set)
        elif isinstance(encodings, dict):
            # 2.0.0 format: dict keyed by tensor name
            for old_name in list(encodings.keys()):
                new_name = _map_base_to_peft_name(old_name, target_set)
                if new_name != old_name:
                    encodings[new_name] = encodings.pop(old_name)

    return adapted


def _map_base_to_peft_name(name: str, target_modules: set[str]) -> str:
    """Transform a single base model encoding name to PeftModel name.

    Handles two naming conventions:

    Slash paths (activations):
        ``/model/layers.0/self_attn/q_proj/Conv_output_0``
        → ``/base_model/model/layers.0/self_attn/q_proj/base_layer/Conv_output_0``

    Dotted paths (params):
        ``model.layers.0.self_attn.q_proj.weight``
        → ``base_model.model.layers.0.self_attn.q_proj.base_layer.weight``
    """
    if "/" in name:
        # Slash path (activation encodings)
        for target in target_modules:
            marker = f"/{target}/"
            if marker in name:
                name = name.replace(marker, f"/{target}/base_layer/", 1)
                break
        if name.startswith("/model/"):
            name = "/base_model" + name
    else:
        # Dotted path (param encodings)
        for target in target_modules:
            marker = f".{target}."
            if marker in name:
                name = name.replace(marker, f".{target}.base_layer.", 1)
                break
        if name.startswith("model."):
            name = "base_model." + name
        elif not name.startswith("base_model."):
            # lm_head.weight → base_model.lm_head.weight? No — check PeftModel naming
            # PeftModel: /base_model/lm_head/Conv, so lm_head is NOT under model.
            name = "base_model." + name

    return name


# ---------------------------------------------------------------------------
# Recipe exclusion helpers
# ---------------------------------------------------------------------------


def get_lora_node_names_for_exclusion(
    model: "onnx.ModelProto",
    lora_names: dict,
) -> list[str]:
    """Return ONNX node names in LoRA branches for recipe exclusion.

    Traces downstream from LoRA weight initializers and scale constants,
    collecting all nodes that are exclusively part of LoRA computation.
    Stops at Add/Sum merge points where the LoRA branch rejoins the base
    path (one input from LoRA, one from base) — those are NOT excluded
    since they carry the base computation forward.

    The returned names match ConnectedGraph op names (which are ONNX node
    names), suitable for passing as ``nodes_to_exclude`` to SeqMSE and
    AdaScale.

    :param model: ONNX ModelProto (raw or from QuantSim ``sim.model.model``).
    :param lora_names: Dict as returned by ``prepare_lora_onnx()``.
    :return: List of ONNX node names in LoRA branches.
    """
    lora_param_set = set(lora_names["params"])
    scale_set = set(lora_names.get("scales", {}))
    # Non-LoRA initializers are treated as constants (not traced through).
    # LoRA params and scales are excluded from the const set so the trace
    # propagates through them and their QcQuantizeOp wrappers.
    all_init_names = {init.name for init in model.graph.initializer}
    const_names = all_init_names - lora_param_set - scale_set
    lora_tensors = lora_param_set | scale_set

    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    lora_node_names: set[str] = set()
    frontier = list(lora_tensors)
    visited: set[str] = set()

    while frontier:
        next_frontier = []
        for tensor_name in frontier:
            for node in consumers.get(tensor_name, []):
                if node.name in visited:
                    continue
                visited.add(node.name)

                non_const_inputs = [
                    inp for inp in node.input if inp and inp not in const_names
                ]
                any_lora = any(inp in lora_tensors for inp in non_const_inputs)
                if not any_lora:
                    continue

                all_lora = all(inp in lora_tensors for inp in non_const_inputs)

                # Stop at merge points: Add/Sum where base and LoRA paths meet
                if node.op_type in ("Add", "Sum") and not all_lora:
                    continue

                lora_node_names.add(node.name)
                for out in node.output:
                    if out:
                        lora_tensors.add(out)
                        next_frontier.append(out)
        frontier = next_frontier

    logger.info(
        "Found %d LoRA branch node names for recipe exclusion",
        len(lora_node_names),
    )
    return sorted(lora_node_names)


# ---------------------------------------------------------------------------
# Phase lifecycle helpers
# ---------------------------------------------------------------------------


def disable_lora_quantizers(sim, lora_names: dict[str, list[str]]) -> int:
    """Disable all LoRA quantizers (Phase 2: recipe compatibility).

    During recipe application (AdaScale, SeqMSE, LPBQ), LoRA quantizers
    should be disabled so they don't interfere with base-model optimization.

    :param sim: QuantizationSimModel instance
    :param lora_names: Dict with ``"params"``, ``"activations"``, ``"scales"`` keys
    :return: Number of quantizers disabled
    """
    all_lora = lora_names["params"] + lora_names["activations"]
    count = 0
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = False
            count += 1

    # Scales are already disabled by set_lora_bitwidth, but ensure it
    for name in lora_names.get("scales", {}):
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = False

    logger.info("Disabled %d LoRA quantizers for recipe compatibility", count)
    return count


def enable_lora_calibration(
    sim,
    lora_names: dict[str, list[str]],
    param_type: str | qtype,
    activation_type: str | qtype,
) -> int:
    """Enable LoRA quantizers and set bitwidth for calibration (Phase 3).

    Enables all LoRA quantizers and sets their bitwidth. LoRA weights remain
    as initializers (not promoted to ``graph.input``). Use ``set_lora_weights()``
    to swap adapter weights before ``compute_encodings()``.

    :param sim: QuantizationSimModel instance.
    :param lora_names: Dict with ``"params"``, ``"activations"``, ``"scales"`` keys.
    :param param_type: Quantization type for LoRA weights (e.g., ``"int16"``).
    :param activation_type: Quantization type for LoRA activations (e.g., ``"int8"``).
    :return: Number of quantizers enabled.
    """
    all_lora = lora_names["params"] + lora_names["activations"]
    count = 0
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = True
            count += 1

    set_lora_bitwidth(sim, lora_names, param_type, activation_type)

    logger.info("Enabled %d LoRA quantizers for calibration", count)
    return count


def export_for_deployment(
    sim,
    lora_names: dict[str, list[str]],
    adapter_encodings: dict[str, dict[str, dict]],
    output_dir: str,
    filename_prefix: str,
) -> None:
    """Write QAIRT deployment artifacts (Phase 4).

    Exports base model encodings and per-adapter LoRA encodings, plus QAIRT
    config files (lora_weight_list, lora_config, adapter list).

    :param sim: QuantizationSimModel instance.
    :param lora_names: Dict with ``"params"``, ``"activations"``, ``"scales"`` keys.
    :param adapter_encodings: Dict mapping adapter name to encoding dict
        (from ``get_lora_encodings()``).
    :param output_dir: Directory to write deployment artifacts.
    :param filename_prefix: Prefix for output filenames.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sim.export(str(output_path), filename_prefix)

    for adapter_name, encodings in adapter_encodings.items():
        encoding_path = output_path / f"{filename_prefix}_{adapter_name}.encodings"
        with open(encoding_path, "w", encoding="utf-8") as f:
            json.dump(encodings, f, indent=2)

    adapter_names = list(adapter_encodings.keys())
    write_lora_weight_list(lora_names, str(output_path / "lora_weight_list.txt"))
    write_lora_config(adapter_names, str(output_path), filename_prefix)
    write_adapter_list(adapter_names, str(output_path), filename_prefix)

    logger.info(
        "Exported deployment artifacts for %d adapters to %s",
        len(adapter_names),
        output_dir,
    )


def load_adapter(
    sim,
    lora_names: dict,
    adapter_name: str,
    weights: dict[str, np.ndarray],
    encodings: dict[str, dict],
    param_type: str | qtype = "int16",
    activation_type: str | qtype = "int8",
) -> None:
    """Atomically load an adapter: weights + encodings + bitwidths.

    Sets LoRA initializer data, quantizer encodings, and bitwidths in one
    call. Prevents mismatched state (adapter A weights with adapter B
    encodings). Handles safetensors key remapping internally.

    After this call, ``sim.session.run(None, inputs)`` will use the loaded
    adapter without requiring adapter weights in the feed dict.

    :param sim: QuantizationSimModel instance.
    :param lora_names: Dict as returned by ``prepare_lora_onnx()``.
    :param adapter_name: Name of the adapter to load (e.g., ``"code"``).
    :param weights: Raw safetensors weights dict (from
        ``safetensors.numpy.load_file()``).
    :param encodings: Quantizer encodings for this adapter (from
        ``get_lora_encodings()``).
    :param param_type: Quantization type for LoRA weights (default ``"int16"``).
    :param activation_type: Quantization type for LoRA activations
        (default ``"int8"``).
    """
    from onnx import numpy_helper as np_helper

    is_multi = "adapters" in lora_names
    model_proto = sim.model.model

    # Step 1: Remap safetensors keys → ONNX init names (with transposition)
    all_param_names = set(lora_names["params"])
    tp = lora_names.get("transposed_params")
    remapped = _remap_safetensors_to_onnx(
        weights,
        adapter_name,
        is_multi,
        onnx_param_names=all_param_names,
        transposed_params=tp,
    )

    # Step 2: Update initializer data
    init_map = {init.name: init for init in model_proto.graph.initializer}

    if is_multi:
        adapter_names = get_adapter_names(lora_names)
        for a_name in adapter_names:
            adapter_ln = get_adapter_lora_names(lora_names, a_name)
            for param_name in adapter_ln["params"]:
                if param_name not in init_map:
                    continue
                init = init_map[param_name]
                if a_name == adapter_name and param_name in remapped:
                    new_data = remapped[param_name]
                else:
                    shape = tuple(init.dims)
                    new_data = np.zeros(shape, dtype=np.float32)
                new_init = np_helper.from_array(new_data, name=param_name)
                init.CopyFrom(new_init)
    else:
        for param_name in lora_names["params"]:
            if param_name not in init_map:
                continue
            init = init_map[param_name]
            if param_name in remapped:
                new_data = remapped[param_name]
            else:
                shape = tuple(init.dims)
                new_data = np.zeros(shape, dtype=np.float32)
            new_init = np_helper.from_array(new_data, name=param_name)
            init.CopyFrom(new_init)

    # Step 3: Set quantizer encodings
    set_lora_encodings(sim, encodings)

    # Step 4: Set bitwidths
    set_lora_bitwidth(sim, lora_names, param_type, activation_type)

    # Step 5: Rebuild session to pick up new initializer data
    sim._rebuild_session()  # pylint: disable=protected-access

    logger.info(
        "Loaded adapter '%s' (%d weights, %d encodings)",
        adapter_name,
        len(remapped),
        len(encodings),
    )


def set_lora_weights(
    sim,
    lora_names: dict,
    adapter_name: str,
    weights: dict[str, np.ndarray],
) -> None:
    """Set LoRA initializer data for a specific adapter and rebuild session.

    Unlike ``load_adapter()``, this does NOT set encodings or bitwidths.
    Use during calibration where ``compute_encodings()`` determines encodings.

    :param sim: QuantizationSimModel instance.
    :param lora_names: Dict as returned by ``prepare_lora_onnx()``.
    :param adapter_name: Name of the adapter to activate.
    :param weights: Raw safetensors weights dict.
    """
    from onnx import numpy_helper as np_helper

    is_multi = "adapters" in lora_names
    model_proto = sim.model.model
    all_param_names = set(lora_names["params"])
    tp = lora_names.get("transposed_params")
    remapped = _remap_safetensors_to_onnx(
        weights,
        adapter_name,
        is_multi,
        onnx_param_names=all_param_names,
        transposed_params=tp,
    )

    init_map = {init.name: init for init in model_proto.graph.initializer}

    if is_multi:
        # Multi-adapter: set target adapter's weights, zero all others
        adapter_names = get_adapter_names(lora_names)
        for a_name in adapter_names:
            adapter_ln = get_adapter_lora_names(lora_names, a_name)
            for param_name in adapter_ln["params"]:
                if param_name not in init_map:
                    continue
                init = init_map[param_name]
                if a_name == adapter_name and param_name in remapped:
                    new_data = remapped[param_name]
                else:
                    shape = tuple(init.dims)
                    new_data = np.zeros(shape, dtype=np.float32)
                new_init = np_helper.from_array(new_data, name=param_name)
                init.CopyFrom(new_init)
    else:
        # Single-adapter: set all LoRA params from remapped weights
        for param_name in lora_names["params"]:
            if param_name not in init_map:
                continue
            init = init_map[param_name]
            if param_name in remapped:
                new_data = remapped[param_name]
            else:
                shape = tuple(init.dims)
                new_data = np.zeros(shape, dtype=np.float32)
            new_init = np_helper.from_array(new_data, name=param_name)
            init.CopyFrom(new_init)

    sim._rebuild_session()  # pylint: disable=protected-access
    logger.info(
        "Set LoRA weights for adapter '%s' (%d weights)", adapter_name, len(remapped)
    )


# ---------------------------------------------------------------------------
# QAIRT artifact helpers
# ---------------------------------------------------------------------------


def write_lora_weight_list(lora_names: dict[str, list[str]], path: str) -> None:
    """Write LoRA tensor names for ``qairt-converter --lora_weight_list``.

    :param lora_names: Dict with ``"params"`` and ``"activations"`` keys
    :param path: Output file path
    """
    sorted_names = sorted(lora_names["params"], key=_natural_sort_key)
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted_names:
            f.write(name + "\n")
    logger.info("Wrote %d LoRA tensor names to %s", len(sorted_names), path)


def write_lora_config(
    adapter_names: list[str],
    output_dir: str,
    filename_prefix: str,
    model_name: str | None = None,
) -> None:
    """Write ``lora_config.yaml`` for ``qairt-lora-importer --lora_config``.

    :param adapter_names: List of adapter names
    :param output_dir: Directory to write the config
    :param filename_prefix: Model filename prefix
    :param model_name: ONNX model filename. Default: ``{filename_prefix}.onnx``
    """
    if model_name is None:
        model_name = f"{filename_prefix}.onnx"

    lines = ["use_case:"]
    for adapter_name in sorted(adapter_names):
        lines.append(f'  - name: "{adapter_name}"')
        lines.append(f'    model_name: "{model_name}"')
        lines.append(f'    lora_weights: "{adapter_name}.safetensors"')
        lines.append(
            f'    quant_overrides: "{filename_prefix}_{adapter_name}.encodings"'
        )
        lines.append('    output_path: "./output"')

    config_path = Path(output_dir) / "lora_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote lora_config.yaml with %d use-cases", len(adapter_names))


def write_adapter_list(
    adapter_names: list[str],
    output_dir: str,
    filename_prefix: str,
) -> None:
    """Write ``lora_adaptor_list.yaml`` for MHA2SHA ``--lora-adaptor-list``.

    .. note:: The YAML filename uses British spelling (``adaptor``) for MHA2SHA compatibility.

    :param adapter_names: List of adapter names
    :param output_dir: Directory to write the config
    :param filename_prefix: Model filename prefix
    """
    lines = []
    for adapter_name in sorted(adapter_names):
        lines.append(f'- name: "{adapter_name}"')
        lines.append(f'  encodings_path: "{filename_prefix}_{adapter_name}.encodings"')
        lines.append(f'  safetensor_path: "{adapter_name}.safetensors"')

    config_path = Path(output_dir) / "lora_adaptor_list.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote lora_adaptor_list.yaml with %d adapters", len(adapter_names))
