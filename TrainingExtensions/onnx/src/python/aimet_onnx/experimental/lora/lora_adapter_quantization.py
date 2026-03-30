# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LoRA quantization utilities for integrating with QuantizationSimModel.

Standalone functions that interact with QuantSim's qc_quantize_op_dict to
configure, freeze, and unfreeze quantizers for multi-adapter calibration
workflows. All functions take ``lora_names: dict`` with ``"params"`` and
``"activations"`` keys, as returned by ``export_peft_to_onnx()``.
"""

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


def _validate_lora_names(lora_names: dict[str, list[str]]) -> None:
    """Validate the lora_names dict has the expected structure."""
    if "params" not in lora_names or "activations" not in lora_names:
        raise ValueError(
            "lora_names must have 'params' and 'activations' keys, "
            f"got keys: {list(lora_names.keys())}"
        )
    if not lora_names["params"] and not lora_names["activations"]:
        raise ValueError(
            "lora_names has empty 'params' and 'activations' lists. "
            "Ensure export_peft_to_onnx() found LoRA weights in the model."
        )


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
    _validate_lora_names(lora_names)

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

    logger.info(
        "Set LoRA bitwidth: %d quantizers (params=%d-bit, activations=%d-bit)",
        count,
        param_bw,
        activation_bw,
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
    _validate_lora_names(lora_names)
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
    _validate_lora_names(lora_names)
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
    _validate_lora_names(lora_names)
    all_lora = lora_names["params"] + lora_names["activations"]
    count = 0
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            qtzr._is_encoding_frozen = False  # pylint: disable=protected-access
            qtzr.reset_encoding_stats()
            count += 1

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
    _validate_lora_names(lora_names)
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
    _validate_lora_names(lora_names)
    init_map = {init.name: init for init in model.graph.initializer}
    param_names = lora_names["params"]

    missing = [name for name in param_names if name not in init_map]
    if missing:
        raise ValueError(
            f"{len(missing)} LoRA names not found in model initializers: {missing[:5]}. "
            f"Ensure the model was exported with export_peft_to_onnx()."
        )

    zero_weights = {}
    for name in param_names:
        init = init_map[name]
        shape = tuple(init.dims)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        zero_weights[name] = np.zeros(shape, dtype=dtype)

    return zero_weights


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


def write_adaptor_list(
    adapter_names: list[str],
    output_dir: str,
    filename_prefix: str,
) -> None:
    """Write ``lora_adaptor_list.yaml`` for MHA2SHA ``--lora-adaptor-list``.

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
    logger.info("Wrote lora_adaptor_list.yaml with %d adaptors", len(adapter_names))
