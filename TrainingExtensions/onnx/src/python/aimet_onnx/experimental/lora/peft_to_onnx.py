# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch-to-ONNX export for LoRA models.

Provides ``export_peft_to_onnx()`` which handles the full pipeline: take a
user-loaded PeftModel, export to ONNX with LoRA weights as initializers,
then prepare the model for adapter swapping.

Requires PyTorch + PEFT.
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx

from aimet_onnx.experimental.lora.lora_adapter_quantization import LoRAResult

logger = logging.getLogger(__name__)


def export_peft_to_onnx(
    peft_model,
    sample_inputs: tuple,
    adapter_paths: Dict[str, str],
    output_dir: str,
    *,
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_shapes: dict = None,
    opset_version: int = 18,
    model_filename: str = "model.onnx",
) -> Tuple[onnx.ModelProto, LoRAResult]:
    """Export a PeftModel with LoRA adapters to ONNX.

    The user loads their own model and provides sample inputs — this function
    handles ONNX export and LoRA preparation. Follows the same pattern as
    ``QuantizationSimModel(model, dummy_input)``.

    :param peft_model: A PEFT model with at least one LoRA adapter applied.
        The user is responsible for loading the base model and applying adapters.
    :param sample_inputs: Tuple of sample input tensors for tracing, like
        ``dummy_input`` for QuantSim. Example: ``(input_ids,)`` for LLMs,
        ``(sample, timestep, encoder_hidden_states)`` for UNet.
    :param adapter_paths: Dict mapping adapter name to HuggingFace ID or local path
        for each PEFT adapter.
    :param output_dir: Directory to save the ONNX model and adapter files.
    :param input_names: ONNX input names. If None, inferred from the base model's
        forward signature.
    :param output_names: ONNX output names. Default: ``["output"]``.
    :param dynamic_shapes: Dynamic shape specification for ``torch.onnx.export``.
        If None, batch dimension (dim 0) is made dynamic for all inputs.
    :param opset_version: ONNX opset version (default 18).
    :param model_filename: Filename for the exported ONNX model (default "model.onnx").
    :return: Tuple of ``(model, result)`` — the ONNX ModelProto with LoRA weights
        as initializers, and a LoRAResult with adapter data.
    """
    try:
        import torch  # noqa: F401  # pylint: disable=unused-import
    except ImportError as e:
        raise ImportError(
            "export_peft_to_onnx requires torch and peft. "
            "Install with: pip install torch peft"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer defaults if not provided
    if input_names is None:
        input_names = _infer_input_names(peft_model, sample_inputs)
    if output_names is None:
        output_names = ["output"]
    if dynamic_shapes is None:
        dynamic_shapes = _infer_dynamic_shapes(sample_inputs, input_names)

    # Extract PEFT adapter config
    if not peft_model.peft_config:
        raise ValueError(
            "PeftModel has no adapter configs. Ensure the model was loaded "
            "with at least one PEFT adapter."
        )
    adapter_key = list(peft_model.peft_config.keys())[0]
    peft_config = peft_model.peft_config[adapter_key]

    # Export to ONNX
    onnx_path = str(output_dir / model_filename)
    logger.info("Exporting to ONNX: %s", onnx_path)

    _export_model_to_onnx(
        peft_model,
        onnx_path,
        opset_version,
        sample_inputs,
        input_names,
        output_names,
        dynamic_shapes,
    )

    # Collect LoRA initializer names from the PeftModel. dynamo=True with
    # optimize=False guarantees ONNX names match PyTorch parameter names.
    # _Wrapper stores PeftModel as self.model, adding a "model." prefix.
    lora_initializer_names = [
        f"model.{name}" for name, _ in peft_model.named_parameters() if "lora_" in name
    ]

    # Save adapter_config.json
    config_dict = {
        "r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "target_modules": list(peft_config.target_modules),
        "lora_dropout": peft_config.lora_dropout,
        "bias": peft_config.bias,
        "peft_type": "LORA",
    }
    config_path = str(output_dir / "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load ONNX model and prepare for LoRA. Dynamo export preserves parameter
    # names, so we know exactly which ONNX initializers are LoRA weights.
    model_proto = onnx.load(onnx_path)
    init_map = {init.name: init for init in model_proto.graph.initializer}

    lora_inits = []
    for name in lora_initializer_names:
        if name not in init_map:
            raise ValueError(f"Initializer '{name}' not found in model")
        lora_inits.append(init_map[name])

    # Extract default adapter weights and cache shapes
    default_weights = {}
    lora_shapes = {}
    for init in lora_inits:
        default_weights[init.name] = onnx.numpy_helper.to_array(init)
        shape = tuple(init.dims)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        lora_shapes[init.name] = (shape, dtype)

    lora_input_names = [init.name for init in lora_inits]

    result = LoRAResult(
        adapters={"default": default_weights},
        lora_input_names=lora_input_names,
        lora_shapes=lora_shapes,
    )

    # Load adapter weights from safetensors files on disk
    for adapter_name, adapter_path in adapter_paths.items():
        logger.info("Loading adapter '%s' from: %s", adapter_name, adapter_path)
        weights = _load_adapter_safetensors(adapter_path, result.lora_input_names)
        result.adapters[adapter_name] = weights

    # Clean up external data files from the dynamo save before re-saving
    # with a consistent filename. Dynamo's save may use a different external
    # data filename than ours.
    _cleanup_external_data(output_dir, model_filename, keep=set())

    # Always save with external data. LLMs exceed protobuf's 2 GB serialization limit,
    # and ByteSize() itself fails with EncodeError on protobuf 5+.
    # After save, reload tensor data into the in-memory proto (save strips it).
    onnx.save_model(
        model_proto,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{model_filename}.data",
    )
    onnx.load_external_data_for_model(model_proto, str(output_dir))
    logger.info("Saved prepared model to: %s", onnx_path)
    result.model_path = str(onnx_path)

    return model_proto, result


def _export_model_to_onnx(
    model,
    onnx_path: str,
    opset_version: int,
    sample_inputs: tuple,
    input_names: List[str],
    output_names: List[str],
    dynamic_shapes: dict,
) -> None:
    """Export a PeftModel to ONNX via ``torch.onnx.export(dynamo=True)``.

    Wraps the model in ``_Wrapper`` which:

    - Stores PeftModel as ``self.model``, adding a ``model.`` prefix to all
      ONNX parameter names (needed for name-based LoRA detection).
    - Calls ``base_model(...)`` to bypass ``PeftModel.forward()``'s
      adapter context-manager hooks.
    - Extracts the primary output tensor from structured outputs
      (e.g. ``CausalLMOutput.logits``).
    - Uses dynamically generated ``forward()`` with explicitly named
      parameters so dynamo export preserves input names in the ONNX graph.

    .. note::
        ``optimize=False`` is required — the ONNX graph optimizer renames
        initializers to anonymous names, breaking the LoRA name matching.
    """
    import torch

    class _Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

    # Dynamically build forward() with named parameters matching input_names.
    # Dynamo export derives ONNX input names from the function signature.
    param_list = ", ".join(input_names)
    forward_src = (
        f"def forward(self, {param_list}):\n"
        f"    out = self.model.base_model({param_list})\n"
        f"    return _extract_output(out)\n"
    )
    exec_ns = {"_extract_output": _extract_output}
    exec(forward_src, exec_ns)  # noqa: S102  # pylint: disable=exec-used
    _Wrapper.forward = exec_ns["forward"]

    # Build dynamic_shapes as a dict keyed by parameter names
    if dynamic_shapes:
        dynamic_shapes_dict = dict(zip(input_names, dynamic_shapes))
    else:
        dynamic_shapes_dict = None

    wrapped = _Wrapper(model)
    wrapped.eval()

    onnx_program = torch.onnx.export(
        wrapped,
        sample_inputs,
        dynamo=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes_dict,
        opset_version=opset_version,
        optimize=False,
    )

    onnx_program.save(onnx_path, external_data=True)


def _extract_output(out):
    """Extract a single tensor from structured model output.

    HuggingFace models return structured outputs (CausalLMOutput, UNet2DOutput,
    etc.) rather than raw tensors. This extracts the primary output tensor.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("logits", "sample", "last_hidden_state"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            return getattr(out, attr)
    return out[0]


def _infer_input_names(peft_model, sample_inputs: tuple) -> List[str]:
    """Infer ONNX input names from the base model's forward signature.

    First tries required (no-default) parameters. If none are found (common
    with HuggingFace models where all params have defaults like
    ``input_ids: Optional[...] = None``), uses the first N positional
    parameter names matching ``len(sample_inputs)``.
    """
    base = peft_model.get_base_model()
    sig = inspect.signature(base.forward)

    positional_names = [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    # Try required-only first
    required = [
        name
        for name in positional_names
        if sig.parameters[name].default is inspect.Parameter.empty
    ]
    if required and len(required) >= len(sample_inputs):
        return required[: len(sample_inputs)]

    # HuggingFace models: all params have defaults. Use positional order.
    if len(positional_names) >= len(sample_inputs):
        return positional_names[: len(sample_inputs)]

    return [f"input_{i}" for i in range(len(sample_inputs))]


def _infer_dynamic_shapes(sample_inputs: tuple, input_names: List[str]) -> tuple:  # pylint: disable=unused-argument
    """Make batch dimension (dim 0) dynamic for all inputs with dim > 0.

    Returns a tuple of per-input shape specs, converted to a dict keyed by
    parameter name in ``_export_model_to_onnx`` before passing to dynamo.
    """
    import torch

    batch = torch.export.Dim("batch")
    return tuple({0: batch} if t.dim() > 0 else {} for t in sample_inputs)


def _load_adapter_safetensors(
    adapter_path: str,
    onnx_input_names: List[str],
) -> Dict[str, np.ndarray]:
    """Load LoRA weights from safetensors and map to ONNX input names.

    Reads ``adapter_model.safetensors`` directly from disk — no PyTorch
    model needed. Supports local directories and HuggingFace Hub IDs.

    Name mapping is deterministic, based on ``dynamo=True`` export guarantees::

        ONNX:        model.base_model.model...lora_A.default.weight
        Safetensors:       base_model.model...lora_A.weight

    Two differences (both reversed by :func:`_onnx_name_to_safetensors_key`):

    1. ONNX has ``model.`` prefix (``_Wrapper`` stores PeftModel as ``self.model``)
    2. ONNX includes adapter name (``.default.``) between ``lora_A``/``lora_B``
       and ``weight``; PEFT strips this when saving to safetensors

    :param adapter_path: Local directory or HuggingFace Hub ID
    :param onnx_input_names: ONNX graph input names for LoRA weights
    :return: Dict mapping ONNX input name to numpy weight array
    """
    from safetensors.numpy import load_file

    local_path = Path(adapter_path)

    if local_path.is_dir():
        sf_file = local_path / "adapter_model.safetensors"
        if not sf_file.exists():
            raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_path}")
        raw_weights = load_file(str(sf_file))
    elif local_path.is_file():
        raw_weights = load_file(str(local_path))
    else:
        # HuggingFace Hub ID — download the safetensors file
        from huggingface_hub import hf_hub_download

        sf_file = hf_hub_download(adapter_path, "adapter_model.safetensors")
        raw_weights = load_file(sf_file)

    # Map each ONNX input name to its safetensors key, then look up the weight
    mapped = {}
    for onnx_name in onnx_input_names:
        sf_key = _onnx_name_to_safetensors_key(onnx_name)
        if sf_key in raw_weights:
            mapped[onnx_name] = raw_weights[sf_key]

    if len(mapped) < len(onnx_input_names):
        unmapped = [n for n in onnx_input_names if n not in mapped]
        raise ValueError(
            f"Only mapped {len(mapped)}/{len(onnx_input_names)} ONNX inputs "
            f"from safetensors at '{adapter_path}'. "
            f"Unmapped: {unmapped[:5]}. "
            f"This indicates a name mismatch between the ONNX model and the adapter."
        )

    return mapped


def _onnx_name_to_safetensors_key(onnx_name: str) -> str:
    """Convert an ONNX initializer name to the corresponding safetensors key.

    Reverses two naming differences introduced during export:

    1. Strips ``model.`` prefix added by ``_Wrapper``
    2. Removes the single-segment adapter name (e.g. ``default``, ``style``)
       between ``lora_A``/``lora_B`` and ``weight``. PEFT's
       ``named_parameters()`` includes this segment, but PEFT strips it when
       saving to safetensors.

    .. note::
        This assumes PEFT adapter names are single dot-separated segments
        (e.g. ``default``, ``style``). Multi-segment adapter names containing
        dots would cause incorrect stripping.
    """
    # Strip _Wrapper's "model." prefix
    key = onnx_name.split(".", 1)[1] if onnx_name.startswith("model.") else onnx_name

    # Strip adapter name between lora_A/B and weight:
    #   "...lora_A.default.weight" → "...lora_A.weight"
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part in ("lora_A", "lora_B") and i + 2 < len(parts):
            parts = parts[: i + 1] + parts[i + 2 :]
            break

    return ".".join(parts)


def _cleanup_external_data(output_dir: Path, model_filename: str, keep: set) -> None:
    """Remove external data files left by a previous ONNX save.

    Dynamo's ``onnx_program.save()`` creates external data files with names
    that may differ from the final ``onnx.save_model()`` call. This removes
    any ``*.data`` files associated with the model, except those in *keep*.

    :param output_dir: Directory containing the ONNX model
    :param model_filename: ONNX model filename (e.g. ``"model.onnx"``)
    :param keep: Set of filenames to preserve
    """
    for data_file in output_dir.glob(f"{model_filename}*data*"):
        if data_file.name not in keep:
            data_file.unlink(missing_ok=True)
