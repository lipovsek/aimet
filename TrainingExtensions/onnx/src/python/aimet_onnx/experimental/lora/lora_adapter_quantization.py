# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LoRA quantization utilities for integrating with QuantizationSimModel.

Standalone functions that interact with QuantSim's qc_quantize_op_dict to
configure, freeze, and unfreeze quantizers for multi-adapter calibration
workflows. All functions take ``lora_names: dict`` with ``"params"``,
``"activations"``, and ``"scales"`` keys, as returned by ``configure_lora_onnx()``.
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

    ``"base_model.model.layers.0.self_attn.q_proj.lora_scale"`` → ``"q_proj"``
    """
    parts = scale_name.replace(".lora_scale", "").split(".")
    return parts[-1] if parts else ""


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
