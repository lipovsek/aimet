# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Configure user-exported ONNX models for LoRA quantization.

Provides ``configure_lora_onnx()`` to match and rename LoRA initializers,
convert scale constants, register LoRA weights as overridable graph inputs,
and trace activation tensors.  Also provides ``add_lora_branches()`` to
insert LoRA branches for additional adapter target modules.
"""

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from aimet_onnx.quantsim import op_outputs_to_ignore

logger = logging.getLogger(__name__)

_PEFT_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# configure_lora_onnx — primary public API
# ---------------------------------------------------------------------------


def configure_lora_onnx(
    onnx_path: str | Path,
    peft_keys: Iterable[str],
    output_path: str | Path,
    adapter_paths: list[str] = None,
    *,
    adapter_name: str = "default",
) -> tuple[onnx.ModelProto, dict]:
    """Configure a user-exported ONNX model for LoRA quantization.

    Takes an ONNX model and PEFT state dict keys, and returns a model
    with LoRA weights registered as overridable graph inputs along with
    the names needed for ``QuantizationSimModel`` setup.

    The ONNX model must be exported with ``dynamo=True`` and
    ``optimize=False`` to preserve LoRA initializer names::

        torch.onnx.export(
            peft_model.base_model.model, sample_inputs, "model.onnx",
            dynamo=True, optimize=False,
        )

    :param onnx_path: Path to user-exported ONNX model.
    :param peft_keys: PEFT state dict key strings.  Can come from
        ``get_peft_model_state_dict(peft_model).keys()`` or
        ``safetensors.numpy.load_file("adapter_model.safetensors").keys()``.
    :param output_path: Where to save the configured model.
    :param adapter_paths: List of adapter directory paths.  Each directory
        must contain an ``adapter_config.json``.  Branches are inserted for
        any target modules not already present in the exported graph.
        Pass ``[]`` or ``None`` for single-adapter models.
    :param adapter_name: PEFT adapter name segment in ONNX initializer
        names (e.g. ``"default"``).  Stripped during name matching.
    :return: ``(model, lora_names)`` where ``lora_names`` has ``"params"``,
        ``"activations"``, and ``"scales"`` keys.
    """
    if adapter_paths is None:
        adapter_paths = []

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = output_path.name

    safetensors_keys = set(peft_keys)

    model_proto = onnx.load(str(onnx_path), load_external_data=False)
    init_map = {init.name: init for init in model_proto.graph.initializer}

    # Index PEFT keys by suffix after "base_model.model." for matching
    # against ONNX initializer names (which omit this prefix).
    suffix_to_peft_key = {}
    for key in safetensors_keys:
        if key.startswith(_PEFT_PREFIX):
            suffix_to_peft_key[key[len(_PEFT_PREFIX) :]] = key

    rename_map = {}
    lora_param_names = []
    for name in init_map:
        parts = [p for p in name.split(".") if p != adapter_name]
        stripped = ".".join(parts)
        if stripped in suffix_to_peft_key:
            rename_map[name] = suffix_to_peft_key[stripped]
            lora_param_names.append(name)
        elif stripped.endswith(".weight"):
            # Try progressively shorter suffixes to handle varying prefixes
            segments = stripped.split(".")
            for i in range(1, len(segments)):
                suffix = ".".join(segments[i:])
                if suffix in suffix_to_peft_key:
                    rename_map[name] = suffix_to_peft_key[suffix]
                    lora_param_names.append(name)
                    break

    if not lora_param_names:
        raise ValueError(
            "No LoRA initializers found in ONNX model matching provided keys. "
            "Ensure the model was exported with dynamo=True and optimize=False: "
            "torch.onnx.export(model, inputs, path, dynamo=True, optimize=False)"
        )

    lora_param_names.sort()

    _rename_lora_initializers(model_proto, rename_map)
    lora_param_names = [rename_map[name] for name in lora_param_names]

    # Convert shared scale Constant nodes to per-branch named initializers
    scales = _convert_lora_scales(model_proto, lora_param_names)

    # Register LoRA initializers as graph inputs so weights can be overridden via feed dict
    _dual_list_lora_initializers(model_proto, lora_param_names)

    # Insert branches for additional adapter target modules
    for adapter_path in adapter_paths:
        config_path = Path(adapter_path) / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)

        new_names = add_lora_branches(model_proto, adapter_config)
        lora_param_names.extend(new_names["params"])
        scales.update(new_names["scales"])

    lora_param_names.sort()
    lora_activation_names = _trace_lora_activations(model_proto, lora_param_names)

    lora_names = {
        "params": lora_param_names,
        "activations": lora_activation_names,
        "scales": scales,
    }

    onnx.load_external_data_for_model(model_proto, str(onnx_path.parent))
    _cleanup_external_data(output_dir, model_filename)

    onnx.save_model(
        model_proto,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{model_filename}.data",
    )
    onnx.load_external_data_for_model(model_proto, str(output_dir))
    logger.info("Saved prepared model to: %s", output_path)

    return model_proto, lora_names


# ---------------------------------------------------------------------------
# add_lora_branches — public API for inserting LoRA branches
# ---------------------------------------------------------------------------


def add_lora_branches(
    model: onnx.ModelProto,
    adapter_config: dict | str | Path,
    *,
    default_rank: int = 8,
) -> dict:
    """Insert LoRA branches for target modules not already in the graph.

    For each target module in the adapter config that does not already have
    a LoRA branch, this function inserts a
    ``Gemm(lora_A) → Gemm(lora_B) → Mul(scale) → Add`` subgraph that
    intercepts the base module's output.

    :param model: ONNX ModelProto (modified in-place).
    :param adapter_config: Either a dict (parsed adapter_config.json) or a
        path to ``adapter_config.json``.
    :param default_rank: Default LoRA rank if not specified in config.
    :return: ``lora_names`` dict with ``"params"``, ``"activations"``, and
        ``"scales"`` keys for the **newly inserted** branches only.
    """
    if isinstance(adapter_config, (str, Path)):
        with open(adapter_config, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)

    target_modules = adapter_config.get("target_modules", [])
    rank = adapter_config.get("r", default_rank)
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    lora_alpha = adapter_config.get("lora_alpha", float(rank))
    scale_value = float(lora_alpha) / float(rank)

    if not target_modules:
        logger.warning("adapter_config has no target_modules — no branches inserted")
        return {"params": [], "activations": [], "scales": {}}

    # Build lookup: initializer name → TensorProto
    init_map = {init.name: init for init in model.graph.initializer}
    existing_inputs = {inp.name for inp in model.graph.input}

    # Find which target modules already have LoRA branches
    existing_lora_modules = set()
    for name in init_map:
        if "lora_A" in name or "lora_a" in name.lower():
            for tm in target_modules:
                if f".{tm}." in name:
                    existing_lora_modules.add(tm)

    # Build consumer map: tensor name → list of nodes that consume it
    consumer_map = {}
    for node in model.graph.node:
        for inp in node.input:
            consumer_map.setdefault(inp, []).append(node)

    new_params = []
    new_scales = {}

    for tm in target_modules:
        if tm in existing_lora_modules:
            logger.debug("Skipping %s — LoRA branch already exists", tm)
            continue

        # Find base weight initializers matching this target module
        base_weight_names = _find_base_weights_for_module(init_map, tm)
        if not base_weight_names:
            logger.warning("No base weight found for target module '%s' — skipping", tm)
            continue

        for base_weight_name in base_weight_names:
            _insert_lora_branch(
                model,
                base_weight_name,
                rank,
                scale_value,
                init_map,
                existing_inputs,
                consumer_map,
                new_params,
                new_scales,
            )

    new_activations = _trace_lora_activations(model, new_params) if new_params else []

    return {
        "params": sorted(new_params),
        "activations": new_activations,
        "scales": new_scales,
    }


# ---------------------------------------------------------------------------
# Private helpers — branch insertion
# ---------------------------------------------------------------------------


def _find_base_weights_for_module(init_map: dict, target_module: str) -> list[str]:
    """Find base weight initializer names for a target module.

    Matches names ending with ``.{target_module}.weight`` or
    ``.{target_module}.base_layer.weight``, or exact matches like
    ``{target_module}.weight`` for top-level modules.
    """
    matches = []
    candidates = (
        f".{target_module}.weight",
        f".{target_module}.base_layer.weight",
        f"{target_module}.weight",
        f"{target_module}.base_layer.weight",
    )
    for name in init_map:
        if any(name.endswith(s) for s in candidates[:2]) or name in candidates[2:]:
            # Skip if this is already a LoRA weight (not a base weight)
            if "lora_A" in name or "lora_B" in name:
                continue
            matches.append(name)
    matches.sort()
    return matches


def _find_linear_consumer(
    base_weight_name: str,
    consumer_map: dict,
) -> tuple | None:
    """Find the Gemm/MatMul node consuming a base weight.

    Detects both direct (weight → Gemm/MatMul) and indirect
    (weight → Transpose → MatMul) patterns.

    :return: ``(linear_node, gemm_x, uses_transpose)`` or ``None``.
    """
    base_consumers = consumer_map.get(base_weight_name, [])
    transpose_weight_output = None

    for node in base_consumers:
        if node.op_type in ("Gemm", "MatMul"):
            gemm_x = _non_weight_input(node, {base_weight_name})
            return (node, gemm_x, False) if gemm_x else None
        if node.op_type == "Transpose":
            transpose_weight_output = node.output[0]
            for matmul in consumer_map.get(transpose_weight_output, []):
                if matmul.op_type == "MatMul":
                    weight_inputs = {base_weight_name, transpose_weight_output}
                    gemm_x = _non_weight_input(matmul, weight_inputs)
                    return (matmul, gemm_x, True) if gemm_x else None
    return None


def _non_weight_input(node, weight_inputs: set) -> str | None:
    """Return the first non-weight, non-empty input of a node."""
    for inp in node.input:
        if inp and inp not in weight_inputs:
            return inp
    return None


def _create_lora_initializers(
    model: onnx.ModelProto,
    lora_a_name: str,
    lora_b_name: str,
    scale_name: str,
    in_features: int,
    out_features: int,
    rank: int,
    scale_value: float,
    init_map: dict,
    existing_inputs: set,
) -> None:
    """Create lora_A, lora_B, and scale initializers and register them as graph inputs."""
    for name, shape in [
        (lora_a_name, (rank, in_features)),
        (lora_b_name, (out_features, rank)),
    ]:
        data = np.zeros(shape, dtype=np.float32)
        init = numpy_helper.from_array(data, name=name)
        model.graph.initializer.append(init)
        init_map[name] = init
        _add_graph_input_for_lora(model, name, init, existing_inputs)

    scale_init = numpy_helper.from_array(
        np.array(scale_value, dtype=np.float32), name=scale_name
    )
    model.graph.initializer.append(scale_init)
    init_map[scale_name] = scale_init
    _add_graph_input_for_lora(model, scale_name, scale_init, existing_inputs)


def _build_lora_subgraph(
    model: onnx.ModelProto,
    linear_node,
    gemm_x: str,
    lora_a_name: str,
    lora_b_name: str,
    scale_name: str,
    uses_transpose: bool,
) -> None:
    """Build the LoRA branch nodes and wire them into the graph.

    Redirects the base linear node's output through an ``Add`` that
    combines it with ``lora_A → lora_B → scale``.
    """
    original_output = linear_node.output[0]
    base_output_renamed = f"{original_output}_base"
    linear_node.output[0] = base_output_renamed

    suffix = lora_a_name.replace(".lora_A.weight", "").replace(".", "_")
    lora_a_out = f"lora_a_out_{suffix}"
    lora_b_out = f"lora_b_out_{suffix}"
    lora_scaled_out = f"lora_scaled_{suffix}"

    if uses_transpose:
        lora_a_t = f"lora_a_transposed_{suffix}"
        lora_b_t = f"lora_b_transposed_{suffix}"

        lora_nodes = [
            helper.make_node(
                "Transpose",
                [lora_a_name],
                [lora_a_t],
                name=f"lora_transpose_a_{suffix}",
            ),
            helper.make_node(
                "MatMul",
                [gemm_x, lora_a_t],
                [lora_a_out],
                name=f"lora_matmul_a_{suffix}",
            ),
            helper.make_node(
                "Transpose",
                [lora_b_name],
                [lora_b_t],
                name=f"lora_transpose_b_{suffix}",
            ),
            helper.make_node(
                "MatMul",
                [lora_a_out, lora_b_t],
                [lora_b_out],
                name=f"lora_matmul_b_{suffix}",
            ),
        ]
    else:
        lora_nodes = [
            helper.make_node(
                "Gemm",
                [gemm_x, lora_a_name],
                [lora_a_out],
                transB=1,
                name=f"lora_gemm_a_{suffix}",
            ),
            helper.make_node(
                "Gemm",
                [lora_a_out, lora_b_name],
                [lora_b_out],
                transB=1,
                name=f"lora_gemm_b_{suffix}",
            ),
        ]

    mul_node = helper.make_node(
        "Mul",
        [lora_b_out, scale_name],
        [lora_scaled_out],
        name=f"lora_mul_{suffix}",
    )
    add_node = helper.make_node(
        "Add",
        [base_output_renamed, lora_scaled_out],
        [original_output],
        name=f"lora_add_{suffix}",
    )
    model.graph.node.extend([*lora_nodes, mul_node, add_node])


def _insert_lora_branch(
    model: onnx.ModelProto,
    base_weight_name: str,
    rank: int,
    scale_value: float,
    init_map: dict,
    existing_inputs: set,
    consumer_map: dict,
    new_params: list,
    new_scales: dict,
) -> None:
    """Insert a single LoRA branch for one base weight."""
    base_init = init_map[base_weight_name]
    out_features = base_init.dims[0]
    in_features = base_init.dims[1] if len(base_init.dims) > 1 else base_init.dims[0]

    # Derive LoRA param names from base weight name
    base_prefix = base_weight_name
    for suffix in (".weight", ".base_layer.weight"):
        if base_prefix.endswith(suffix):
            base_prefix = base_prefix[: -len(suffix)]
            break
    if not base_prefix.startswith(_PEFT_PREFIX):
        base_prefix = f"{_PEFT_PREFIX}{base_prefix}"

    lora_a_name = f"{base_prefix}.lora_A.weight"
    lora_b_name = f"{base_prefix}.lora_B.weight"
    scale_name = f"{base_prefix}.lora_scale"

    if lora_a_name in init_map:
        return

    result = _find_linear_consumer(base_weight_name, consumer_map)
    if result is None:
        logger.warning(
            "No Gemm/MatMul node found consuming %s — cannot insert LoRA branch",
            base_weight_name,
        )
        return
    linear_node, gemm_x, uses_transpose = result

    _create_lora_initializers(
        model,
        lora_a_name,
        lora_b_name,
        scale_name,
        in_features,
        out_features,
        rank,
        scale_value,
        init_map,
        existing_inputs,
    )
    _build_lora_subgraph(
        model,
        linear_node,
        gemm_x,
        lora_a_name,
        lora_b_name,
        scale_name,
        uses_transpose,
    )

    new_params.extend([lora_a_name, lora_b_name])
    new_scales[scale_name] = scale_value
    logger.info(
        "Inserted LoRA branch for %s (rank=%d, scale=%.2f)",
        base_prefix,
        rank,
        scale_value,
    )


def _add_graph_input_for_lora(
    model: onnx.ModelProto,
    name: str,
    init: TensorProto,
    existing_inputs: set,
) -> None:
    """Register a LoRA initializer as a graph input, using ``"lora_rank"`` for rank dims."""
    if name in existing_inputs:
        return

    shape = list(init.dims)
    if "lora_A" in name and len(shape) >= 1:
        shape[0] = "lora_rank"
    elif "lora_B" in name and len(shape) >= 2:
        shape[-1] = "lora_rank"
    # Scale is scalar — no dynamic dims

    value_info = helper.make_tensor_value_info(name, init.data_type, shape)
    model.graph.input.append(value_info)
    existing_inputs.add(name)


# ---------------------------------------------------------------------------
# Private helpers — name matching and graph manipulation
# ---------------------------------------------------------------------------


def _rename_lora_initializers(
    model: onnx.ModelProto, rename_map: dict[str, str]
) -> None:
    """Rename LoRA initializers, graph inputs, and node references in-place."""
    for init in model.graph.initializer:
        if init.name in rename_map:
            init.name = rename_map[init.name]

    for inp in model.graph.input:
        if inp.name in rename_map:
            inp.name = rename_map[inp.name]

    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in rename_map:
                node.input[i] = rename_map[name]
        for i, name in enumerate(node.output):
            if name in rename_map:
                node.output[i] = rename_map[name]


def _convert_lora_scales(
    model: onnx.ModelProto, lora_param_names: list[str]
) -> dict[str, float]:
    """Replace shared Constant scale nodes with named initializers.

    PEFT export produces ``alpha/rank`` as anonymous Constant nodes.
    This replaces each with a named initializer registered as a graph
    input, so the scale value can be overridden per adapter at runtime.

    :return: Dict mapping scale initializer name to its float value.
    """
    # Find lora_B param names
    lora_b_names = [n for n in lora_param_names if "lora_B" in n]
    if not lora_b_names:
        return {}

    # Build node lookup maps
    consumer_map = {}
    for node in model.graph.node:
        for inp in node.input:
            consumer_map.setdefault(inp, []).append(node)

    producer_map = {}
    for node in model.graph.node:
        for out in node.output:
            producer_map[out] = node

    scales = {}
    constants_to_remove = set()
    existing_inputs = {inp.name for inp in model.graph.input}
    init_names = {init.name for init in model.graph.initializer}

    for lora_b_name in lora_b_names:
        # Trace: lora_B weight → (Transpose →) Gemm/MatMul → output → Mul
        consumers = consumer_map.get(lora_b_name, [])
        linear_output = None

        # Direct: lora_B → Gemm/MatMul
        for n in consumers:
            if n.op_type in ("Gemm", "MatMul"):
                linear_output = n.output[0]
                break

        # Indirect: lora_B → Transpose → MatMul
        if linear_output is None:
            for n in consumers:
                if n.op_type == "Transpose":
                    for n2 in consumer_map.get(n.output[0], []):
                        if n2.op_type == "MatMul":
                            linear_output = n2.output[0]
                            break
                    if linear_output:
                        break

        if linear_output is None:
            continue

        mul_nodes = [
            n for n in consumer_map.get(linear_output, []) if n.op_type == "Mul"
        ]
        if not mul_nodes:
            continue

        mul_node = mul_nodes[0]

        # Find the Constant input to the Mul (the scale)
        scale_input = None
        scale_value = None
        for inp in mul_node.input:
            if inp == linear_output:
                continue
            # Check if this input comes from a Constant node
            if inp in producer_map and producer_map[inp].op_type == "Constant":
                const_node = producer_map[inp]
                for attr in const_node.attribute:
                    if attr.name == "value":
                        scale_value = float(numpy_helper.to_array(attr.t))
                scale_input = inp
                constants_to_remove.add(id(const_node))
                break
            # Already a named initializer — skip
            if inp in init_names and "lora_scale" in inp:
                scale_input = None
                break

        if scale_input is None or scale_value is None:
            continue

        # Derive scale name from lora_B name
        # "base_model.model...lora_B.weight" → "base_model.model...lora_scale"
        base_prefix = lora_b_name.replace(".lora_B.weight", "")
        scale_name = f"{base_prefix}.lora_scale"

        # Create named initializer
        scale_data = np.array(scale_value, dtype=np.float32)
        scale_init = numpy_helper.from_array(scale_data, name=scale_name)
        model.graph.initializer.append(scale_init)

        # Add as graph input (scalar, no dynamic dims)
        if scale_name not in existing_inputs:
            value_info = helper.make_tensor_value_info(
                scale_name, TensorProto.FLOAT, []
            )
            model.graph.input.append(value_info)
            existing_inputs.add(scale_name)

        # Update Mul node input to use the named initializer
        for i, inp in enumerate(mul_node.input):
            if inp == scale_input:
                mul_node.input[i] = scale_name

        scales[scale_name] = scale_value

    # Remove Constant nodes that are no longer needed.
    # Only remove if no other node still references the Constant's output.
    # Pre-compute all inputs consumed by non-constant nodes for O(n) check.
    all_consumed_inputs = set()
    for node in model.graph.node:
        if id(node) not in constants_to_remove:
            all_consumed_inputs.update(node.input)

    nodes_to_keep = []
    for node in model.graph.node:
        if id(node) in constants_to_remove:
            if any(out in all_consumed_inputs for out in node.output):
                nodes_to_keep.append(node)
        else:
            nodes_to_keep.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(nodes_to_keep)

    logger.info("Converted %d LoRA scale constants to named initializers", len(scales))
    return scales


def _trace_lora_activations(
    model: onnx.ModelProto, lora_param_names: list[str]
) -> list[str]:
    """Find activation tensor names in LoRA branches for quantizer placement.

    Starting from LoRA weight initializers, traces downstream through
    each LoRA branch (Transpose → MatMul → MatMul → Mul) and collects
    intermediate tensor names.  Stops at the ``Add`` node where the
    LoRA branch merges back into the base path.  Skips ops listed in
    ``op_outputs_to_ignore`` since QuantSim does not quantize those.
    """
    consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    initializer_names = {init.name for init in model.graph.initializer}
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

                non_const_inputs = [
                    inp
                    for inp in node.input
                    if inp and (inp in lora_tensors or inp not in initializer_names)
                ]
                any_lora = any(inp in lora_tensors for inp in non_const_inputs)
                if not any_lora:
                    continue

                # Add is the merge point: base output + LoRA output.
                # Stop only when not ALL inputs are LoRA (= base is present).
                all_lora = all(inp in lora_tensors for inp in non_const_inputs)
                if node.op_type == "Add" and not all_lora:
                    continue

                is_quantized = node.op_type not in op_outputs_to_ignore

                for output in node.output:
                    if (
                        output
                        and output not in lora_tensors
                        and output not in initializer_names
                    ):
                        lora_tensors.add(output)
                        next_frontier.append(output)
                        if is_quantized:
                            activation_names.append(output)
        frontier = next_frontier

    if len(activation_names) < len(lora_param_names) // 2:
        logger.warning(
            "Traced only %d activations for %d LoRA params — "
            "expected at least one activation per LoRA pair",
            len(activation_names),
            len(lora_param_names),
        )

    logger.info("Traced %d LoRA activation tensors", len(activation_names))
    return activation_names


def _dual_list_lora_initializers(model: onnx.ModelProto, lora_names: list[str]) -> None:
    """Register LoRA initializers as graph inputs so they can be overridden via feed dict.

    The initializers stay in ``graph.initializer`` (providing defaults) and are
    also added to ``graph.input``.  Rank dimensions use ``"lora_rank"`` to
    allow adapters with different ranks.
    """
    init_map = {init.name: init for init in model.graph.initializer}
    existing_inputs = {inp.name for inp in model.graph.input}

    for name in lora_names:
        if name in existing_inputs:
            continue
        init = init_map[name]
        _add_graph_input_for_lora(model, name, init, existing_inputs)


def _cleanup_external_data(output_dir: Path, model_filename: str) -> None:
    """Remove existing external data files to avoid conflicts when saving."""
    for data_file in output_dir.glob(f"{model_filename}.data*"):
        data_file.unlink(missing_ok=True)
