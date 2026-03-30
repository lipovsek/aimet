# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch-to-ONNX export for LoRA models.

Provides ``export_peft_to_onnx()`` which exports a PeftModel to ONNX
with LoRA weights prepared for adapter swapping.

Requires PyTorch + PEFT.
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx

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
) -> Tuple[onnx.ModelProto, Dict[str, List[str]]]:
    """Export a PeftModel with LoRA adapters to ONNX.

    :param peft_model: A PEFT model with at least one LoRA adapter applied.
    :param sample_inputs: Tuple of sample input tensors for tracing.
    :param adapter_paths: Dict mapping adapter name to HuggingFace ID or local path
        for each PEFT adapter. Adapter weights are saved as
        ``{output_dir}/{adapter_name}.safetensors``.
    :param output_dir: Directory to save the ONNX model and adapter files.
    :param input_names: ONNX input names. If None, inferred from the base model's
        forward signature.
    :param output_names: ONNX output names. Default: ``["output"]``.
    :param dynamic_shapes: Dynamic shape specification for ``torch.onnx.export``.
        If None, batch dimension (dim 0) is made dynamic for all inputs.
    :param opset_version: ONNX opset version (default 18).
    :param model_filename: Filename for the exported ONNX model (default "model.onnx").
    :return: Tuple of ``(model, lora_names)`` where ``lora_names`` is a dict
        with ``"params"`` (LoRA weight tensor names) and ``"activations"``
        (LoRA branch activation tensor names).
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

    lora_param_names = []
    for name in lora_initializer_names:
        if name not in init_map:
            raise ValueError(f"Initializer '{name}' not found in model")
        lora_param_names.append(name)

    _dual_list_lora_initializers(model_proto, lora_param_names)

    # Trace downstream from LoRA weights to find LoRA branch activations
    lora_activation_names = _trace_lora_activations(model_proto, lora_param_names)

    lora_names = {
        "params": lora_param_names,
        "activations": lora_activation_names,
    }

    # Save adapter weights as safetensors files
    for adapter_name, adapter_path in adapter_paths.items():
        logger.info("Loading adapter '%s' from: %s", adapter_name, adapter_path)
        weights = _load_adapter_safetensors(adapter_path, lora_param_names)
        safetensors_path = str(output_dir / f"{adapter_name}.safetensors")
        _save_adapter_safetensors(weights, safetensors_path)
        logger.info("Saved adapter '%s' to: %s", adapter_name, safetensors_path)

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

    return model_proto, lora_names


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

    Uses ``_Wrapper`` to add ``model.`` prefix to parameter names,
    bypass PeftModel's adapter hooks, and generate a ``forward()`` with
    named parameters so dynamo preserves input names in the ONNX graph.

    ``optimize=False`` is required — the optimizer renames initializers,
    breaking LoRA name matching.
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
    """Extract a single tensor from structured model output."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("logits", "sample", "last_hidden_state"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            return getattr(out, attr)
    return out[0]


def _infer_input_names(peft_model, sample_inputs: tuple) -> List[str]:
    """Infer ONNX input names from the base model's forward signature.

    Tries required parameters first, then falls back to positional order.
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
    """Make batch dimension (dim 0) dynamic for all inputs with dim > 0."""
    import torch

    batch = torch.export.Dim("batch")
    return tuple({0: batch} if t.dim() > 0 else {} for t in sample_inputs)


def _load_adapter_safetensors(
    adapter_path: str,
    onnx_input_names: List[str],
) -> Dict[str, np.ndarray]:
    """Load LoRA weights from safetensors and map to ONNX input names.

    Supports local directories, direct file paths, and HuggingFace Hub IDs.

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

    Strips ``model.`` prefix and removes the adapter name segment between
    ``lora_A``/``lora_B`` and ``weight``. Assumes single-segment adapter names.
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


def _trace_lora_activations(
    model: onnx.ModelProto, lora_param_names: List[str]
) -> List[str]:
    """BFS from LoRA weight tensors to find LoRA branch activation names.

    :param model: ONNX ModelProto
    :param lora_param_names: LoRA weight tensor names (seeds for the trace)
    :return: List of LoRA activation tensor names
    """
    # Build index: tensor name → list of nodes that consume it
    consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    # Also track all initializer names to exclude them from activations
    initializer_names = {init.name for init in model.graph.initializer}

    # BFS from LoRA param names. A node's outputs are LoRA activations if
    # any of its inputs are LoRA-related. We continue tracing past a node
    # only if all its non-constant inputs are LoRA-related (stops at the
    # Add where base and LoRA merge — the Add output IS included but its
    # downstream consumers are not).
    lora_tensors = set(lora_param_names)
    activation_names = []
    frontier = list(lora_param_names)
    visited_nodes = set()

    while frontier:
        next_frontier = []
        for tensor_name in frontier:
            for node in consumers.get(tensor_name, []):
                node_id = id(node)
                if node_id in visited_nodes:
                    continue
                visited_nodes.add(node_id)

                # Add all outputs as LoRA activations
                new_outputs = []
                for output in node.output:
                    if (
                        output
                        and output not in lora_tensors
                        and output not in initializer_names
                    ):
                        lora_tensors.add(output)
                        activation_names.append(output)
                        new_outputs.append(output)

                # Continue tracing only if ALL non-constant inputs are
                # LoRA-related. At the Add (base + LoRA), the base input
                # is not in lora_tensors, so we stop propagating.
                non_const_inputs = [
                    inp for inp in node.input if inp and inp not in initializer_names
                ]
                if all(inp in lora_tensors for inp in non_const_inputs):
                    next_frontier.extend(new_outputs)
        frontier = next_frontier

    logger.info("Traced %d LoRA activation tensors", len(activation_names))
    return activation_names


def _dual_list_lora_initializers(model: onnx.ModelProto, lora_names: List[str]) -> None:
    """Add LoRA initializers to graph.input while keeping them in graph.initializer.

    :param model: ONNX ModelProto (modified in-place)
    :param lora_names: List of LoRA initializer names
    """
    init_map = {init.name: init for init in model.graph.initializer}
    existing_inputs = {inp.name for inp in model.graph.input}

    for name in lora_names:
        if name in existing_inputs:
            continue
        init = init_map[name]
        value_info = onnx.helper.make_tensor_value_info(
            name, init.data_type, list(init.dims)
        )
        model.graph.input.append(value_info)


def _save_adapter_safetensors(weights: Dict[str, np.ndarray], path: str) -> None:
    """Save adapter weights as a safetensors file.

    :param weights: Dict mapping ONNX input name to numpy weight array
    :param path: Output file path
    """
    from safetensors.numpy import save_file

    save_file(weights, path)


def _cleanup_external_data(output_dir: Path, model_filename: str, keep: set) -> None:
    """Remove stale external data files from a previous ONNX save.

    :param output_dir: Directory containing the ONNX model
    :param model_filename: ONNX model filename (e.g. ``"model.onnx"``)
    :param keep: Set of filenames to preserve
    """
    for data_file in output_dir.glob(f"{model_filename}*data*"):
        if data_file.name not in keep:
            data_file.unlink(missing_ok=True)
