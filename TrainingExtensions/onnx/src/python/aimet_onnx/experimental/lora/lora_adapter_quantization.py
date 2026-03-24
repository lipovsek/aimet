# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LoRA quantization utilities for integrating with QuantizationSimModel.

Standalone functions that accept a LoRAResult and interact with QuantSim's
qc_quantize_op_dict to configure, freeze, and unfreeze quantizers for
multi-adapter calibration workflows.
"""

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto

# TODOs
# 1. Need to make exports work with QDQ ONNX


# Default LoRA param type — imported lazily to avoid hard dependency on
# aimet_onnx.common.defs at module load time (e.g., when running unit tests
# without the full AIMET package installed).
_DEFAULT_LORA_PARAM_TYPE = None

logger = logging.getLogger(__name__)


def _natural_sort_key(s: str):
    """Sort key that handles numeric segments naturally.

    Splits the string on digit boundaries so that numeric parts are compared
    as integers. This ensures ``layers.2`` sorts before ``layers.10``.
    """
    return [
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", s)
    ]


@dataclass
class LoRAResult:
    """Result of LoRA model preparation.

    Returned alongside the ``onnx.ModelProto`` by ``export_peft_to_onnx()``.
    """

    adapters: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    lora_input_names: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    lora_shapes: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = field(
        default_factory=dict
    )
    adapter_encodings: Dict[str, Dict] = field(default_factory=dict)

    def get_zero_weights(self) -> Dict[str, np.ndarray]:
        """Return zero-valued weights for all LoRA inputs.

        Used to calibrate the base model with LoRA disabled.

        :return: Dict mapping ONNX input name to zero numpy array
        :raises ValueError: If ``lora_shapes`` is missing entries for any LoRA input
        """
        missing = [n for n in self.lora_input_names if n not in self.lora_shapes]
        if missing:
            raise ValueError(
                f"lora_shapes is missing {len(missing)} entries: {missing[:5]}. "
                f"This usually means LoRAResult was constructed incorrectly."
            )

        zero_weights = {}
        for name in self.lora_input_names:
            shape, dtype = self.lora_shapes[name]
            zero_weights[name] = np.zeros(shape, dtype=dtype)
        return zero_weights

    def get_adapter(self, name: str) -> Dict[str, np.ndarray]:
        """Get a previously loaded adapter by name.

        :param name: Adapter name as passed to ``export_peft_to_onnx()`` or set during preparation
        :return: Dict mapping ONNX graph input name to numpy weight array
        :raises KeyError: If the adapter name is not found
        """
        if name not in self.adapters:
            raise KeyError(
                f"Adapter '{name}' not found. "
                f"Available adapters: {list(self.adapters.keys())}"
            )
        return self.adapters[name]


def _convert_initializers_to_inputs(
    model: onnx.ModelProto, lora_inits: List[TensorProto]
) -> List[str]:
    """Convert LoRA initializers to graph inputs so they can be fed at runtime.

    Removes each initializer from model.graph.initializer and adds a corresponding
    TensorValueInfo to model.graph.input.

    :param model: ONNX ModelProto (modified in-place)
    :param lora_inits: List of LoRA TensorProto initializers
    :return: List of LoRA graph input names
    """
    lora_names = {init.name for init in lora_inits}
    input_names = []

    # Remove from initializer list
    remaining = [
        init for init in model.graph.initializer if init.name not in lora_names
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(remaining)

    # Add as graph inputs
    for init in lora_inits:
        elem_type = init.data_type
        shape = list(init.dims)
        value_info = onnx.helper.make_tensor_value_info(init.name, elem_type, shape)
        model.graph.input.append(value_info)
        input_names.append(init.name)

    return input_names


def configure_lora_quantizers(sim, result: LoRAResult, lora_param_type=None) -> int:
    """Configure LoRA quantizers and convert initializers to graph inputs.

    Performs two operations:

    1. **Set bitwidth**: LoRA weights are typically quantized at higher precision
       (e.g., INT16) than the base model (INT8) to preserve the small low-rank
       adaptations.

    2. **Initializer-to-input conversion**: LoRA weights are kept as initializers
       during ``export_peft_to_onnx()`` so that QuantSim classifies them as
       parameters and assigns per-channel quantization. This function then converts
       them to graph inputs for runtime adapter swapping and rebuilds the ONNX
       Runtime session.

    .. important::
        This function **must** be called **before** ``freeze_base_model()`` or any
        ``freeze_*`` function. ``set_bitwidth()`` is a no-op when encodings are
        frozen, so calling it after freezing will silently fail to change the
        LoRA quantizer bitwidth.

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult from export_peft_to_onnx()
    :param lora_param_type: Quantization type for LoRA weights. Accepts an
        ``aimet_onnx`` qtype (e.g., ``aimet_onnx.int16``) or a plain ``int``
        bitwidth. Default: ``aimet_onnx.int16``.
    :return: Number of quantizers updated
    """
    bitwidth = _resolve_bitwidth(lora_param_type)

    lora_names = set(result.lora_input_names)
    count = 0
    for name in lora_names:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            if qtzr.is_encoding_frozen():
                logger.warning(
                    "LoRA quantizer '%s' is frozen. set_bitwidth() will be a no-op. "
                    "Call configure_lora_quantizers() BEFORE freeze_base_model().",
                    name,
                )
            qtzr.set_bitwidth(bitwidth)
            count += 1
        else:
            logger.debug("LoRA input '%s' not found in qc_quantize_op_dict", name)

    logger.info("Configured %d LoRA quantizers to %d-bit", count, bitwidth)

    # Convert LoRA initializers to graph inputs for runtime adapter swapping
    lora_inits = [
        init for init in sim.model.model.graph.initializer if init.name in lora_names
    ]
    if lora_inits:
        _convert_initializers_to_inputs(sim.model.model, lora_inits)
        # Rebuild ORT session after modifying graph structure. AIMET has no
        # public API for this — _rebuild_session is the internal mechanism.
        sim._rebuild_session()  # pylint: disable=protected-access
        logger.info(
            "Converted %d LoRA initializers to graph inputs and rebuilt session",
            len(lora_inits),
        )

    return count


def _resolve_bitwidth(lora_param_type) -> int:
    """Extract bitwidth from a qtype object, int, or None.

    :param lora_param_type: ``aimet_onnx.int16``, plain ``int``, or ``None``
    :return: Integer bitwidth
    """
    if lora_param_type is None:
        # Default to int16
        global _DEFAULT_LORA_PARAM_TYPE  # pylint: disable=global-statement
        if _DEFAULT_LORA_PARAM_TYPE is None:
            from aimet_onnx import int16

            _DEFAULT_LORA_PARAM_TYPE = int16
        lora_param_type = _DEFAULT_LORA_PARAM_TYPE

    if isinstance(lora_param_type, int):
        return lora_param_type

    # qtype object — use to_legacy_repr() to get (QuantizationDataType, bitwidth)
    if hasattr(lora_param_type, "to_legacy_repr"):
        _, bitwidth = lora_param_type.to_legacy_repr()
        return bitwidth

    raise TypeError(
        f"lora_param_type must be an aimet_onnx qtype (e.g., aimet_onnx.int16), "
        f"an int, or None. Got {type(lora_param_type).__name__}: {lora_param_type}"
    )


def freeze_base_param_quantizers(sim, result: LoRAResult) -> int:
    """Freeze all base (non-LoRA) parameter quantizers.

    After calibrating the base model, call this to lock base param encodings
    so they are shared across all adapter calibrations. Activation quantizers
    remain unfrozen to be recalibrated per adapter.

    This is the recommended default strategy (balanced accuracy/speed).

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult from ``export_peft_to_onnx()``
    :return: Number of quantizers frozen
    """
    lora_names = set(result.lora_input_names)
    count = 0
    for name in sim.param_names:
        if name not in lora_names and name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            if qtzr.enabled:
                qtzr.freeze_encodings()
                count += 1

    logger.info("Froze %d base param quantizers", count)
    return count


def freeze_base_activation_quantizers(sim, result: LoRAResult) -> int:
    """Freeze all base activation quantizers.

    Use in combination with freeze_base_param_quantizers() for the "fast"
    strategy where only LoRA quantizers are recalibrated per adapter.

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult from ``export_peft_to_onnx()``
    :return: Number of quantizers frozen
    """
    lora_names = set(result.lora_input_names)
    count = 0
    for name in sim.activation_names:
        if name not in lora_names and name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            if qtzr.enabled:
                qtzr.freeze_encodings()
                count += 1

    logger.info("Froze %d base activation quantizers", count)
    return count


def freeze_base_model(sim, result: LoRAResult) -> int:
    """Freeze ALL base quantizers (params + activations).

    This is the "fast" strategy: only LoRA quantizers are recalibrated per
    adapter. Fastest but potentially lower accuracy.

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult from ``export_peft_to_onnx()``
    :return: Total number of quantizers frozen
    """
    count = freeze_base_param_quantizers(sim, result)
    count += freeze_base_activation_quantizers(sim, result)
    logger.info("Froze %d total base quantizers", count)
    return count


def unfreeze_lora_quantizers(sim, result: LoRAResult) -> int:
    """Unfreeze LoRA quantizers and reset their encoding stats.

    Call this before calibrating a new adapter so LoRA quantizer encodings
    are recomputed for the new adapter weights.

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult from ``export_peft_to_onnx()``
    :return: Number of quantizers unfrozen
    """
    lora_names = set(result.lora_input_names)
    count = 0
    for name in lora_names:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            # AIMET's public API has no unfreeze — _is_encoding_frozen is the
            # only way to unlock a quantizer. reset_encoding_stats is a no-op
            # when frozen, so we must unfreeze first.
            qtzr._is_encoding_frozen = False  # pylint: disable=protected-access
            qtzr.reset_encoding_stats()
            count += 1

    logger.info("Unfroze %d LoRA quantizers", count)
    return count


def export_lora_weights(
    result: LoRAResult,
    weights: Dict[str, np.ndarray],
    path: str,
) -> None:
    """Save adapter weights as a safetensors file.

    The weights dict should map ONNX graph input names to numpy arrays,
    as returned by ``LoRAResult.get_adapter()``.

    :param result: LoRAResult (used for validation)
    :param weights: Dict mapping ONNX input name to numpy weight array
    :param path: Output file path (should end in .safetensors)
    """
    from safetensors.numpy import save_file

    # Validate that all weight names are known LoRA inputs
    lora_names = set(result.lora_input_names)
    unknown = set(weights.keys()) - lora_names
    if unknown:
        raise ValueError(
            f"Weights contain {len(unknown)} names not in lora_input_names: "
            f"{list(unknown)[:5]}. Ensure weights were produced by "
            f"LoRAResult.get_adapter()."
        )

    save_file(weights, path)
    logger.info("Exported %d LoRA weights to %s", len(weights), path)


def get_lora_encodings(
    sim,
    result: LoRAResult,
    encoding_version: str = "1.0.0",
) -> Dict[str, dict]:
    """Capture LoRA quantizer encodings for the current adapter.

    After calibrating with a specific adapter, call this to snapshot the
    LoRA quantizer encodings. These can later be restored with
    ``set_lora_encodings()`` to switch between adapters without recalibration.

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult from ``export_peft_to_onnx()``
    :param encoding_version: Encoding format version (default ``"1.0.0"``)
    :return: Dict mapping LoRA quantizer name to encoding dict
    """
    lora_names = set(result.lora_input_names)
    encodings = {}
    for name in lora_names:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            encoding = qtzr.export_encodings(encoding_version)
            if encoding is not None:
                encodings[name] = encoding

    logger.info("Captured encodings for %d LoRA quantizers", len(encodings))
    return encodings


def set_lora_encodings(
    sim,
    result: LoRAResult,
    encodings: Dict[str, dict],
) -> int:
    """Restore previously captured LoRA quantizer encodings.

    Loads per-adapter LoRA encodings captured by ``get_lora_encodings()``
    and freezes them, so the quantizers use the exact saved values.

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult from ``export_peft_to_onnx()``
    :param encodings: Dict from ``get_lora_encodings()``
    :return: Number of quantizers updated
    """
    lora_names = set(result.lora_input_names)
    count = 0
    for name, encoding_dict in encodings.items():
        if name not in lora_names:
            logger.warning(
                "Encoding name '%s' is not in lora_input_names. Skipping.", name
            )
            continue
        if name not in sim.qc_quantize_op_dict:
            logger.debug("LoRA input '%s' not found in qc_quantize_op_dict", name)
            continue

        qtzr = sim.qc_quantize_op_dict[name]
        # AIMET has no public API for loading per-quantizer encodings.
        # _load_encodings_dict + freeze_encodings is the internal pattern
        # used by AIMET's own set_and_freeze_param_encodings.
        qtzr._is_encoding_frozen = False  # pylint: disable=protected-access
        qtzr._load_encodings_dict(encoding_dict)  # pylint: disable=protected-access
        qtzr.freeze_encodings()
        count += 1

    logger.info("Restored and froze encodings for %d LoRA quantizers", count)
    return count


def calibrate_lora(
    sim,
    result: LoRAResult,
    dataloader: Iterable,
    lora_param_type=None,
) -> None:
    """Configure LoRA quantizers and calibrate the base model and all adapters.

    This is the recommended single-call API. It performs:

    1. **Configure**: Set LoRA quantizer bitwidth and convert initializers to
       graph inputs (equivalent to ``configure_lora_quantizers()``).
    2. **Base calibration**: Run inference with zeroed LoRA weights to calibrate
       base model quantizers, then freeze base param encodings.
    3. **Per-adapter calibration**: For each adapter in ``result.adapters``
       (excluding ``"default"``), unfreeze LoRA quantizers, calibrate with that
       adapter's weights, and capture LoRA encodings into
       ``result.adapter_encodings``.

    The ``dataloader`` should yield dicts of model inputs (e.g.,
    ``{"input_ids": ...}``). LoRA weights are added automatically.
    TODO: Can we not follow the same pattern that we use for compute_encodings?

    .. note::
        The dataloader is materialized to a list on first use so it can be
        iterated multiple times (once for base + once per adapter). Generators
        and single-use iterators are supported but will be fully consumed into
        memory.

    :param sim: QuantizationSimModel instance
    :param result: LoRAResult with adapters loaded
    :param dataloader: Iterable of feed dicts (model inputs only, no LoRA weights)
    :param lora_param_type: Quantization type for LoRA weights. Accepts an
        ``aimet_onnx`` qtype (e.g., ``aimet_onnx.int16``) or a plain ``int``
        bitwidth. Default: ``aimet_onnx.int16``.
    """
    # Materialize to list so we can iterate multiple times (base + N adapters).
    # Generators/iterators would silently yield nothing on second pass.
    if not isinstance(dataloader, (list, tuple)):
        dataloader = list(dataloader)

    # --- Configure LoRA quantizers ---
    # TODO: Need to move this out - bitwidth settings need to be separate from calibration
    configure_lora_quantizers(sim, result, lora_param_type=lora_param_type)

    # --- Base calibration ---
    zero_weights = result.get_zero_weights()

    def _calibrate_base(session):
        for batch in dataloader:
            feed = dict(batch)
            feed.update(zero_weights)
            session.run(None, feed)

    sim.compute_encodings(_calibrate_base)
    freeze_base_param_quantizers(sim, result)

    # --- Per-adapter calibration ---
    adapter_names = [n for n in result.adapters if n != "default"]
    for adapter_name in adapter_names:
        logger.info("Calibrating adapter '%s'", adapter_name)
        unfreeze_lora_quantizers(sim, result)
        adapter_w = result.get_adapter(adapter_name)

        def _calibrate_adapter(session, w=adapter_w):
            for batch in dataloader:
                feed = dict(batch)
                feed.update(w)
                session.run(None, feed)

        sim.compute_encodings(_calibrate_adapter)

        # TODO: This is not a great design choice, passing in an incomplete
        # result data structure and filling in partially
        result.adapter_encodings[adapter_name] = get_lora_encodings(sim, result)

    logger.info("Calibration complete: base + %d adapters", len(adapter_names))


def export_lora(
    sim,
    result: LoRAResult,
    export_dir: str,
    filename_prefix: str = "model",
    target: str = "qairt",
) -> None:
    """Export quantized model with all artifacts needed for deployment.

    **``target="ort"`` produces:**

    1. ``{prefix}.onnx`` — base model with LoRA as graph inputs
    2. ``{prefix}.encodings`` — base quantization encodings
    3. ``{prefix}_{adapter}.encodings`` — per-adapter LoRA encodings
    4. ``{adapter}.safetensors`` — per-adapter weights (ONNX-named keys)

    **``target="qairt"`` produces everything above plus:**

    5. ``lora_weight_list.txt`` — for ``qairt-converter --lora_weight_list``
    6. ``{prefix}_lora_init.onnx`` — model with LoRA as initializers
    7. ``lora_config.yaml`` — for ``qairt-lora-importer --lora_config``
    8. ``lora_adaptor_list.yaml`` — for MHA2SHA ``--lora-adaptor-list``

    :param sim: QuantizationSimModel (calibrated via ``calibrate_lora``)
    :param result: LoRAResult with ``adapter_encodings`` populated
    :param export_dir: Output directory
    :param filename_prefix: Base filename (default ``"model"``)
    :param target: Deployment target — ``"qairt"`` (default) or ``"ort"``
    """
    if target not in ("qairt", "ort"):
        raise ValueError(f"target must be 'qairt' or 'ort', got '{target}'")

    if target == "qairt" and not result.model_path:
        raise ValueError(
            "target='qairt' requires result.model_path to point to the ONNX model "
            "with LoRA as initializers (needed for qairt-converter/qairt-lora-importer). "
            "Ensure export_peft_to_onnx() was called with an output_dir."
        )

    output_path = Path(export_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1-2. Base model + base encodings
    sim.export(str(output_path), filename_prefix)
    logger.info("Exported base model to %s/%s.onnx", export_dir, filename_prefix)

    # 3. Per-adapter encodings
    for adapter_name, encodings in result.adapter_encodings.items():
        set_lora_encodings(sim, result, encodings)
        adapter_prefix = f"{filename_prefix}_{adapter_name}"
        sim.export(str(output_path), adapter_prefix, export_model=False)
        logger.info("Exported encodings for adapter '%s'", adapter_name)

    # 4. Per-adapter safetensors
    for adapter_name in result.adapter_encodings:
        weights = result.get_adapter(adapter_name)
        safetensors_path = str(output_path / f"{adapter_name}.safetensors")
        export_lora_weights(result, weights, safetensors_path)

    if target == "qairt":
        # 5. LoRA weight list
        _write_lora_weight_list(result, str(output_path / "lora_weight_list.txt"))

        # 6. Initializer model (for QAIRT tools that need LoRA as initializers)
        _copy_initializer_model(result.model_path, output_path, filename_prefix)

        # 7-8. QAIRT config files
        init_model_name = f"{filename_prefix}_lora_init.onnx"
        _write_lora_importer_config(
            result, output_path, filename_prefix, init_model_name
        )
        _write_adaptor_list(result, output_path, filename_prefix)

    logger.info(
        "Export complete (target=%s): model + %d adapter encodings + safetensors",
        target,
        len(result.adapter_encodings),
    )


def _write_lora_weight_list(result: LoRAResult, path: str) -> None:
    """Write LoRA tensor names to a text file for ``qairt-converter --lora_weight_list``.

    :param result: LoRAResult with lora_input_names
    :param path: Output file path
    """
    sorted_names = sorted(result.lora_input_names, key=_natural_sort_key)
    with open(path, "w") as f:
        for name in sorted_names:
            f.write(name + "\n")
    logger.info("Wrote %d LoRA tensor names to %s", len(sorted_names), path)


def _copy_initializer_model(
    model_path: str, output_dir: Path, filename_prefix: str
) -> None:
    """Copy the ONNX model with LoRA as initializers to the export directory.

    QAIRT tools (``qairt-converter``, ``qairt-lora-importer``) need LoRA weights as
    initializers, not graph inputs. This copies the original model from
    ``result.model_path`` with a ``_lora_init`` suffix to distinguish it from
    the QuantSim-exported model.

    Handles external data files (``*.data``) alongside the model.

    :param model_path: Source model path (LoRA as initializers)
    :param output_dir: Destination directory
    :param filename_prefix: Base filename prefix
    """
    src = Path(model_path)
    if not src.exists():
        logger.warning("model_path '%s' does not exist. Skipping copy.", model_path)
        return

    dst_name = f"{filename_prefix}_lora_init.onnx"
    dst = output_dir / dst_name

    shutil.copy2(str(src), str(dst))

    # Copy external data file if it exists
    for ext_data in src.parent.glob(src.name + ".data"):
        shutil.copy2(str(ext_data), str(output_dir / (dst_name + ".data")))

    logger.info("Copied initializer model to %s", dst)


def _write_lora_importer_config(
    result: LoRAResult,
    output_dir: Path,
    filename_prefix: str,
    model_name: str,
) -> None:
    """Write ``lora_config.yaml`` for ``qairt-lora-importer --lora_config``.

    Format::

        use_case:
          - name: "<adapter>"
            model_name: "<model>.onnx"
            lora_weights: "<adapter>.safetensors"
            quant_overrides: "<prefix>_<adapter>.encodings"
            output_path: "./output"

    :param result: LoRAResult with adapter_encodings
    :param output_dir: Directory to write the config
    :param filename_prefix: Model filename prefix
    :param model_name: ONNX model filename (the initializer model)
    """
    lines = ["use_case:"]
    for adapter_name in sorted(result.adapter_encodings.keys()):
        lines.append(f'  - name: "{adapter_name}"')
        lines.append(f'    model_name: "{model_name}"')
        lines.append(f'    lora_weights: "{adapter_name}.safetensors"')
        lines.append(
            f'    quant_overrides: "{filename_prefix}_{adapter_name}.encodings"'
        )
        lines.append('    output_path: "./output"')

    config_path = output_dir / "lora_config.yaml"
    with open(config_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(
        "Wrote lora_config.yaml with %d use-cases", len(result.adapter_encodings)
    )


def _write_adaptor_list(
    result: LoRAResult,
    output_dir: Path,
    filename_prefix: str,
) -> None:
    """Write ``lora_adaptor_list.yaml`` for MHA2SHA ``--lora-adaptor-list``.

    Format::

        - name: "<adapter>"
          encodings_path: "<prefix>_<adapter>.encodings"
          safetensor_path: "<adapter>.safetensors"

    :param result: LoRAResult with adapter_encodings
    :param output_dir: Directory to write the config
    :param filename_prefix: Model filename prefix
    """
    lines = []
    for adapter_name in sorted(result.adapter_encodings.keys()):
        lines.append(f'- name: "{adapter_name}"')
        lines.append(f'  encodings_path: "{filename_prefix}_{adapter_name}.encodings"')
        lines.append(f'  safetensor_path: "{adapter_name}.safetensors"')

    config_path = output_dir / "lora_adaptor_list.yaml"
    with open(config_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(
        "Wrote lora_adaptor_list.yaml with %d adaptors",
        len(result.adapter_encodings),
    )
