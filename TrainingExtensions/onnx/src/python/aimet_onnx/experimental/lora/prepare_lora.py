# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare PeftModel ONNX exports for LoRA quantization.

Provides ``prepare_lora_onnx()`` which takes an ONNX model exported from a
PeftModel and configures it for the phase-based LoRA quantization workflow.
Uses **graph pattern matching** to identify LoRA branches and rename opaque
initializers to PEFT convention.

Handles two complications specific to ``dynamo=False`` exports:

1. **Opaque initializer names**: Linear weights become ``onnx::MatMul_NNN``
   instead of named parameters.  Reconstructed from ONNX node names which
   always encode the module hierarchy.

2. **Transposed weight shapes**: TorchScript bakes ``weight.T`` into the
   initializer data for MatMul compatibility.  This function inserts
   ``Transpose`` nodes so that graph inputs accept PEFT-convention shapes,
   enabling direct feed-dict with safetensors values.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# Ops whose outputs are not quantized — copied from quantsim.py to avoid
# pulling in the full quantsim import chain (which requires C++ libs).
_OP_OUTPUTS_TO_IGNORE = {
    "branch",
    "Flatten",
    "Gather",
    "Reshape",
    "Shape",
    "Unsqueeze",
    "Squeeze",
    "Split",
    "Compress",
    "Tile",
    "Transpose",
    "Identity",
}

logger = logging.getLogger(__name__)

_PEFT_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_lora_onnx(
    onnx_path: str | Path,
    adapter_config: str | Path | dict | list,
    output_path: str | Path,
) -> tuple[onnx.ModelProto, dict]:
    """Configure a PeftModel ONNX export for LoRA quantization.

    Discovers LoRA branch patterns in the ONNX graph, renames opaque
    initializers to PEFT convention, inserts Transpose nodes for safetensors
    shape compatibility, and converts scale constants to named initializers.

    Supports both single-adapter and multi-adapter (PeftMixedModel) exports:

    - **Single adapter**: Pass a single adapter_config path or dict.
    - **Multi-adapter**: Pass a list of adapter configs. The ONNX model must
      have been exported with all adapters active via ``PeftMixedModel``.

    Works with both ``dynamo=False`` and ``dynamo=True`` exports. Supports
    MatMul, Gemm, and Conv ops (for adapted models with Conv2d projections).

    LoRA weights are kept as **initializer-only** (not in ``graph.input``)
    for compatibility with recipes (AdaScale, SeqMSE, LPBQ). Use
    ``enable_lora_calibration()`` to promote them for per-adapter calibration.

    :param onnx_path: Path to ONNX model exported from PeftModel.
    :param adapter_config: One or more adapter configs:
        - Single: path to ``adapter_config.json`` or parsed dict
        - Multi: list of paths/dicts (one per adapter in the export)
    :param output_path: Where to save the configured model.
    :return: ``(model, lora_names)`` where ``lora_names`` has ``"params"``,
        ``"activations"``, ``"scales"`` keys. For multi-adapter, also has
        ``"adapters"`` key with per-adapter breakdown.
    """
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse adapter configs (single or list)
    configs = _parse_adapter_configs(adapter_config)
    union_targets = set()
    for cfg in configs:
        union_targets.update(cfg.get("target_modules", []))
    if not union_targets:
        raise ValueError("adapter_config(s) have no target_modules")

    model = onnx.load(str(onnx_path))

    # Step 1: Find LoRA patterns via topology-based graph traversal
    patterns = _find_lora_patterns(model)
    if not patterns:
        raise ValueError(
            "No LoRA patterns found in ONNX model. Ensure the model was "
            "exported from a PeftModel with adapters active (not a base model)."
        )

    # Step 2: Validate against adapter config target_modules
    found_modules = {p.target_module for p in patterns}
    missing = union_targets - found_modules
    if missing:
        logger.warning(
            "adapter_config target_modules not found in graph: %s. "
            "Found: %s. Ensure all adapters were active during export.",
            missing,
            found_modules,
        )
    extra = found_modules - union_targets
    if extra:
        logger.info(
            "Graph has LoRA branches for modules not in adapter_config: %s",
            extra,
        )

    # Determine if multi-adapter
    adapter_names = sorted({p.adapter_name for p in patterns})
    is_multi_adapter = len(adapter_names) > 1

    # Step 3: Split shared initializers (dynamo=False shares zero-init tensors)
    _split_shared_initializers(model, patterns)

    # Step 4: Rename inits to PEFT convention + insert Transpose for shape compat
    rename_map = {}
    lora_param_names = []
    adapter_param_map: dict[str, list[str]] = {}

    for pattern in patterns:
        peft_prefix = f"{_PEFT_PREFIX}{pattern.module_path}"
        if is_multi_adapter:
            lora_a_peft = f"{peft_prefix}.lora_A.{pattern.adapter_name}.weight"
            lora_b_peft = f"{peft_prefix}.lora_B.{pattern.adapter_name}.weight"
        else:
            lora_a_peft = f"{peft_prefix}.lora_A.weight"
            lora_b_peft = f"{peft_prefix}.lora_B.weight"

        rename_map[pattern.lora_a_init] = lora_a_peft
        rename_map[pattern.lora_b_init] = lora_b_peft
        lora_param_names.extend([lora_a_peft, lora_b_peft])
        adapter_param_map.setdefault(pattern.adapter_name, []).extend(
            [lora_a_peft, lora_b_peft]
        )

    # Record which params need transposing when loading safetensors weights.
    # dynamo=False stores weights in MatMul convention (transposed vs PEFT).
    # Instead of inserting Transpose nodes (which break AdaScale's onnx2torch
    # block conversion), we leave the graph untouched and transpose at load
    # time inside _remap_safetensors_to_onnx.
    transposed_params = _detect_transposed_params(model, patterns)

    # Apply renames to all references
    _rename_initializers(model, rename_map)

    # Update transposed_params keys to use renamed (PEFT) names
    transposed_params = {rename_map.get(k, k): v for k, v in transposed_params.items()}

    lora_param_names.sort()

    # Step 5: Convert scale Constants to named initializers
    scales = _convert_scale_constants(model, patterns, is_multi_adapter)

    # Step 6: Trace LoRA activations
    lora_activation_names = _trace_lora_activations(model, lora_param_names)

    # Step 7: Build lora_names structure
    lora_names = {
        "params": lora_param_names,
        "activations": lora_activation_names,
        "scales": scales,
        "transposed_params": transposed_params,
    }

    if is_multi_adapter:
        adapters_dict = {}
        for adapter_name in adapter_names:
            adapter_params = sorted(adapter_param_map.get(adapter_name, []))
            adapter_scales = {
                k: v for k, v in scales.items() if f".{adapter_name}" in k
            }
            adapter_activations = _trace_lora_activations(model, adapter_params)
            adapters_dict[adapter_name] = {
                "params": adapter_params,
                "activations": adapter_activations,
                "scales": adapter_scales,
            }
        lora_names["adapters"] = adapters_dict

    # Save — use external data only for tensors, not the full model.
    # onnx.save_model(save_as_external_data=True) can convert Constant nodes
    # to initializers and drop them, breaking non-LoRA consumers (e.g., LessOrEqual).
    # Instead, externalize only large tensors (initializers) and keep the graph intact.
    _cleanup_external_data(output_path.parent, output_path.name)
    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=f"{output_path.name}.data",
    )
    onnx.save_model(model, str(output_path))

    logger.info(
        "prepare_lora_onnx: configured %d LoRA branches (%d params, "
        "%d activations, %d scales, %d adapters) -> %s",
        len(patterns),
        len(lora_param_names),
        len(lora_activation_names),
        len(scales),
        len(adapter_names),
        output_path,
    )
    return model, lora_names


def _parse_adapter_configs(
    adapter_config: str | Path | dict | list,
) -> list[dict]:
    """Parse one or more adapter configs into a list of dicts."""
    if isinstance(adapter_config, list):
        configs = []
        for cfg in adapter_config:
            if isinstance(cfg, (str, Path)):
                with open(cfg, "r", encoding="utf-8") as f:
                    configs.append(json.load(f))
            else:
                configs.append(cfg)
        return configs

    if isinstance(adapter_config, (str, Path)):
        with open(adapter_config, "r", encoding="utf-8") as f:
            return [json.load(f)]

    return [adapter_config]


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------

_PASSTHROUGH_OPS = frozenset(
    {
        "Transpose",
        "Squeeze",
        "Unsqueeze",
        "Reshape",
        "Cast",
    }
)

_LORA_COMPUTE_OPS = frozenset({"MatMul", "Gemm", "Conv"})


def _trace_to_compute_op(
    producer_map: dict, tensor_name: str, max_depth: int = 10
) -> tuple[onnx.NodeProto | None, str]:
    """Trace backward through shape manipulation ops to find the compute op.

    PEFT's Conv2d LoRA inserts Transpose/Squeeze/Unsqueeze between the Mul
    and the actual Conv ops.  This helper sees through those passthrough ops
    to reach the underlying MatMul, Gemm, or Conv node.

    For MatMul-based models (no passthrough ops), returns the producer
    immediately on the first step — no behavior change.

    :param producer_map: Map from tensor name to producing node.
    :param tensor_name: Starting tensor to trace backward from.
    :param max_depth: Safety limit to prevent infinite loops.
    :return: ``(compute_node, last_tensor_name)`` or ``(None, tensor_name)``
        if no compute op found within max_depth.
    """
    for _ in range(max_depth):
        prod = producer_map.get(tensor_name)
        if prod is None:
            return None, tensor_name
        if prod.op_type not in _PASSTHROUGH_OPS:
            return prod, tensor_name
        tensor_name = prod.input[0]
    return None, tensor_name


# ---------------------------------------------------------------------------
# Config-driven LoRA detection (not yet wired into prepare_lora_onnx)
# ---------------------------------------------------------------------------


def _find_lora_patterns_by_config(
    model: onnx.ModelProto, adapter_configs: list[dict]
) -> list["_LoRAPattern"]:
    """Find LoRA patterns using target_modules from adapter configs as attach points.

    Instead of topology matching (Add ← Mul ← Conv/MatMul chains), uses the
    ``target_modules`` from adapter configs to find LoRA branches by **name**.
    For each target module, scans node names for ``/{target}/{adapter_name}/``
    patterns and finds the compute nodes (Conv/MatMul/Gemm) that consume LoRA
    weight initializers.

    Works for all cases:
    - Non-adapted models (MatMul-based LoRA)
    - Adapted models (Conv2d-based LoRA with passthrough ops)
    - Both ``dynamo=False`` and ``dynamo=True`` exports
    """
    union_targets = set()
    for cfg in adapter_configs:
        union_targets.update(cfg.get("target_modules", []))

    if not union_targets:
        return []

    init_names = {init.name for init in model.graph.initializer}
    producer_map = {out: node for node in model.graph.node for out in node.output}

    # Group nodes by (target_module, adapter_segment)
    # node name pattern: /.../{target_module}/{segment}/{op}
    # segment is one of: base_layer, {adapter_name}, {adapter_name}_N
    module_adapter_nodes: dict[str, dict[str, list]] = {}

    for node in model.graph.node:
        for target in union_targets:
            marker = f"/{target}/"
            if marker not in node.name:
                continue

            # Extract the segment after /{target}/
            parts_after = node.name.split(marker, 1)[1]
            segment = parts_after.split("/")[0]

            # Skip base_layer — that's the base, not LoRA
            if segment == "base_layer":
                continue

            # Strip _N suffix for adapter name (lora_B uses {adapter}_1, etc.)
            adapter = re.sub(r"_\d+$", "", segment)

            # Extract full module path (including layer index)
            # e.g., /base_model/model/layers.0/self_attn/q_proj/tldr/Conv
            # → module_path = layers.0.self_attn.q_proj
            module_path = _extract_module_path_from_lora_node(node.name)
            if module_path is None:
                module_path = target  # fallback

            key = module_path
            module_adapter_nodes.setdefault(key, {}).setdefault(adapter, []).append(
                node
            )

    # Build patterns from grouped nodes
    patterns = []
    for module_path, adapters in module_adapter_nodes.items():
        target_module = module_path.split(".")[-1]

        # Find the base node for this module
        base_node = _find_base_node_by_name(model, module_path)

        for adapter_name, nodes in adapters.items():
            # Find compute nodes (Conv/MatMul/Gemm) that consume initializers
            # (also handles Identity-shared inits from dynamo=False)
            compute_with_init = []
            for n in nodes:
                if n.op_type not in _LORA_COMPUTE_OPS:
                    continue
                init_input = _find_init_input(n, init_names, producer_map)
                if init_input is not None:
                    compute_with_init.append((n, init_input))

            if len(compute_with_init) < 2:
                continue  # Need both lora_A and lora_B

            # First = lora_A, second = lora_B (graph order)
            lora_a_node, lora_a_init = compute_with_init[0]
            lora_b_node, lora_b_init = compute_with_init[1]

            # Find scale: Mul node at /{module_path}/Mul or /{module_path}/Mul_N
            scale_input, scale_value = _find_scale_by_name(
                model, module_path, init_names
            )

            patterns.append(
                _LoRAPattern(
                    module_path=module_path,
                    target_module=target_module,
                    adapter_name=adapter_name,
                    add_node=None,  # Not needed for config-driven
                    mul_node=None,
                    lora_b_matmul=lora_b_node,
                    lora_a_matmul=lora_a_node,
                    base_node=base_node,
                    lora_a_init=lora_a_init,
                    lora_b_init=lora_b_init,
                    lora_b_actual_init=lora_b_init,
                    scale_input=scale_input or "",
                    scale_value=scale_value,
                )
            )

    patterns.sort(key=lambda p: (p.module_path, p.adapter_name))
    logger.info(
        "Config-driven: found %d LoRA patterns (%d adapters) for targets %s",
        len(patterns),
        len({p.adapter_name for p in patterns}),
        union_targets,
    )
    return patterns


def _find_init_input(
    node: onnx.NodeProto, init_names: set, producer_map: dict | None = None
) -> str | None:
    """Find the first initializer name consumed by a node.

    Also handles Identity-shared initializers (dynamo=False shares zero-init
    tensors via Identity nodes).
    """
    for inp in node.input:
        if inp in init_names:
            return inp
        # Check if input comes from an Identity whose input is an initializer
        if producer_map is not None:
            prod = producer_map.get(inp)
            if prod is not None and prod.op_type == "Identity":
                if prod.input[0] in init_names:
                    return inp  # Return the Identity output name (consumed by node)
    return None


def _find_base_node_by_name(
    model: onnx.ModelProto, module_path: str
) -> onnx.NodeProto | None:
    """Find the base_layer compute node for a module by scanning node names."""
    target = module_path.split(".")[-1]
    for node in model.graph.node:
        if f"/{target}/base_layer/" in node.name and node.op_type in _LORA_COMPUTE_OPS:
            # Verify this is the right layer by checking full module path
            extracted = _extract_module_path(node.name)
            if extracted and extracted.endswith(module_path.split(".")[-1]):
                return node
    return None


def _find_scale_by_name(
    model: onnx.ModelProto,
    module_path: str,
    init_names: set,
) -> tuple[str | None, float | None]:
    """Find the scale initializer or Constant for a LoRA branch by node name.

    Scales are Mul nodes at ``/{module}/Mul`` (first adapter) or
    ``/{module}/Mul_N`` (Nth adapter). Each Mul has one input from the
    LoRA branch and one from a Constant or initializer (the scale value).
    """
    target = module_path.split(".")[-1]
    for node in model.graph.node:
        if node.op_type != "Mul":
            continue
        if f"/{target}/Mul" not in node.name:
            continue
        # Check if this Mul's inputs include a scale (Constant or init)
        for inp in node.input:
            if inp in init_names:
                # Already a named initializer — read value
                for init in model.graph.initializer:
                    if init.name == inp:
                        val = float(numpy_helper.to_array(init))
                        return inp, val
                return inp, None

        # Check for Constant producer
        producer_map = {}
        for n in model.graph.node:
            for out in n.output:
                producer_map[out] = n

        for inp in node.input:
            prod = producer_map.get(inp)
            if prod is not None and prod.op_type == "Constant":
                for attr in prod.attribute:
                    if attr.name == "value":
                        val = float(numpy_helper.to_array(attr.t))
                        return inp, val
    return None, None


# ---------------------------------------------------------------------------
# LoRA pattern detection (topology-based — primary path)
# ---------------------------------------------------------------------------


@dataclass
class _LoRAPattern:
    """A discovered LoRA branch in the ONNX graph."""

    module_path: str  # e.g., "attn.q_proj"
    target_module: str  # e.g., "q_proj" (last segment)
    adapter_name: str  # e.g., "code", "medical", "default"
    add_node: onnx.NodeProto
    mul_node: onnx.NodeProto
    lora_b_matmul: onnx.NodeProto
    lora_a_matmul: onnx.NodeProto
    base_node: onnx.NodeProto | None  # None for chained (Pattern B)
    lora_a_init: str  # opaque init name
    lora_b_init: str  # opaque init name (may be via Identity)
    lora_b_actual_init: str  # the real init (before Identity)
    scale_input: str  # Constant output name feeding Mul
    scale_value: float | None = None
    is_chained: bool = False  # True for Pattern B (chained Add)
    chain_parent_add: onnx.NodeProto | None = None  # the Add this chains from


def _find_lora_patterns(model: onnx.ModelProto) -> list[_LoRAPattern]:
    """Find all LoRA branch patterns in the ONNX graph.

    Two-pass detection supporting single-adapter and multi-adapter exports:

    - **Pass 1 (Pattern A)**: Finds Add nodes where one input is a base_layer
      op and the other is ``Mul <- lora_B <- lora_A`` (LoRA branch).
    - **Pass 2 (Pattern B)**: Finds chained Add nodes where one input is a
      previously-matched LoRA Add (for 2nd, 3rd, ... adapters on same module).
      Repeats until no new matches found (supports 3+ adapter chains).

    Works with both ``dynamo=False`` (node-name based) and ``dynamo=True``
    (init-name based) exports. Supports MatMul, Gemm, and Conv ops in LoRA
    branches (for adapted models with Conv2d projections).

    Returns patterns sorted by (module_path, adapter_name) for deterministic
    ordering.
    """
    producer_map = {}
    for node in model.graph.node:
        for out in node.output:
            producer_map[out] = node

    init_names = {init.name for init in model.graph.initializer}

    # Pass 1: Pattern A — base_layer + LoRA branch
    patterns = []
    matched_add_outputs = set()

    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue

        pattern = _try_match_lora_add(node, producer_map, init_names)
        if pattern is not None:
            patterns.append(pattern)
            for out in node.output:
                matched_add_outputs.add(out)

    # Pass 2: Pattern B — chained Adds (repeat until stable)
    found_new = True
    while found_new:
        found_new = False
        for node in model.graph.node:
            if node.op_type != "Add" or len(node.input) != 2:
                continue
            # Skip already-matched nodes
            if any(out in matched_add_outputs for out in node.output):
                continue

            pattern = _try_match_lora_add_chained(
                node, producer_map, init_names, matched_add_outputs
            )
            if pattern is not None:
                patterns.append(pattern)
                for out in node.output:
                    matched_add_outputs.add(out)
                found_new = True

    patterns.sort(key=lambda p: (p.module_path, p.adapter_name))
    logger.info(
        "Found %d LoRA patterns in graph (%d adapters)",
        len(patterns),
        len({p.adapter_name for p in patterns}),
    )
    return patterns


def _try_match_lora_add(
    add_node: onnx.NodeProto,
    producer_map: dict,
    init_names: set,
) -> _LoRAPattern | None:
    """Try to match an Add node as a LoRA merge point.

    Expected pattern::

        Add(
            base_layer/MatMul_output,              # base path
            Mul(                                    # LoRA path
                default_1/MatMul(                   # lora_B
                    default/MatMul(activation, A),  # lora_A
                    B
                ),
                Constant(scale)
            )
        )
    """
    input_a, input_b = add_node.input[0], add_node.input[1]
    producer_a = producer_map.get(input_a)
    producer_b = producer_map.get(input_b)

    if producer_a is None or producer_b is None:
        return None

    # One input should be Mul (LoRA path), other is base path
    if producer_a.op_type == "Mul":
        mul_node = producer_a
        base_input = input_b
    elif producer_b.op_type == "Mul":
        mul_node = producer_b
        base_input = input_a
    else:
        return None

    # Base path: trace through passthrough ops (Squeeze/Transpose/Cast)
    # to find the actual compute node (MatMul/Gemm/Conv)
    base_node, _ = _trace_to_compute_op(producer_map, base_input)
    if base_node is None or base_node.op_type not in _LORA_COMPUTE_OPS:
        return None

    # Verify base node name contains "base_layer"
    if "base_layer" not in base_node.name:
        return None

    # Trace Mul inputs: one is the lora_B output, other is the scale Constant
    mul_lora_input = None
    scale_input = None
    scale_value = None
    for inp in mul_node.input:
        prod = producer_map.get(inp)
        if prod is not None and prod.op_type == "Constant":
            scale_input = inp
            for attr in prod.attribute:
                if attr.name == "value":
                    scale_value = float(numpy_helper.to_array(attr.t))
        elif inp in init_names:
            # Scale could also be an initializer
            scale_input = inp
        elif mul_lora_input is None:
            # Trace through passthrough ops to find lora_B compute node
            compute_op, _ = _trace_to_compute_op(producer_map, inp)
            if compute_op is not None and compute_op.op_type in _LORA_COMPUTE_OPS:
                mul_lora_input = inp

    if mul_lora_input is None or scale_input is None:
        return None

    # lora_B: trace through passthrough ops to the actual compute node
    lora_b_matmul, _ = _trace_to_compute_op(producer_map, mul_lora_input)
    if lora_b_matmul is None or lora_b_matmul.op_type not in _LORA_COMPUTE_OPS:
        return None

    # Find lora_B's weight init (could be direct or via Identity)
    lora_b_init, lora_b_actual_init = _find_weight_init(
        lora_b_matmul, init_names, producer_map
    )
    if lora_b_init is None:
        return None

    # lora_A: trace through passthrough ops from lora_B's non-weight input
    lora_a_matmul = None
    for inp in lora_b_matmul.input:
        if inp == lora_b_init:
            continue
        compute_op, _ = _trace_to_compute_op(producer_map, inp)
        if compute_op is not None and compute_op.op_type in _LORA_COMPUTE_OPS:
            lora_a_matmul = compute_op
            break
    if lora_a_matmul is None:
        return None

    # Find lora_A's weight init
    lora_a_init, _ = _find_weight_init(lora_a_matmul, init_names, producer_map)
    if lora_a_init is None:
        return None

    # Extract module path from Add node name
    module_path = _extract_module_path(add_node.name)
    if module_path is None:
        return None

    target_module = module_path.split(".")[-1]

    # Extract adapter name from lora_A node name or init name
    adapter_name = _extract_adapter_name(lora_a_matmul.name, lora_a_init, init_names)

    return _LoRAPattern(
        module_path=module_path,
        target_module=target_module,
        adapter_name=adapter_name,
        add_node=add_node,
        mul_node=mul_node,
        lora_b_matmul=lora_b_matmul,
        lora_a_matmul=lora_a_matmul,
        base_node=base_node,
        lora_a_init=lora_a_init,
        lora_b_init=lora_b_init,
        lora_b_actual_init=lora_b_actual_init,
        scale_input=scale_input,
        scale_value=scale_value,
    )


def _try_match_lora_add_chained(
    add_node: onnx.NodeProto,
    producer_map: dict,
    init_names: set,
    matched_add_outputs: set,
) -> _LoRAPattern | None:
    """Try to match an Add node as a chained LoRA merge point (Pattern B).

    Pattern B: The non-LoRA input is the output of a previously-matched
    LoRA Add (rather than a base_layer MatMul). This occurs for 2nd, 3rd, ...
    adapters in a multi-adapter chain::

        Add(
            previous_LoRA_Add_output,              # chained from prior adapter
            Mul(                                    # LoRA path (same structure)
                lora_B/MatMul(lora_A/MatMul(x, A), B),
                scale
            )
        )
    """
    input_a, input_b = add_node.input[0], add_node.input[1]
    producer_a = producer_map.get(input_a)
    producer_b = producer_map.get(input_b)

    if producer_a is None or producer_b is None:
        return None

    # One input should be Mul (LoRA path), other should be a matched Add output
    if producer_a.op_type == "Mul" and input_b in matched_add_outputs:
        mul_node = producer_a
        chain_parent_add = producer_map.get(input_b)
    elif producer_b.op_type == "Mul" and input_a in matched_add_outputs:
        mul_node = producer_b
        chain_parent_add = producer_map.get(input_a)
    else:
        return None

    # Trace Mul inputs: one is the lora_B output, other is the scale
    mul_lora_input = None
    scale_input = None
    scale_value = None
    for inp in mul_node.input:
        prod = producer_map.get(inp)
        if prod is not None and prod.op_type == "Constant":
            scale_input = inp
            for attr in prod.attribute:
                if attr.name == "value":
                    scale_value = float(numpy_helper.to_array(attr.t))
        elif inp in init_names:
            scale_input = inp
        elif mul_lora_input is None:
            compute_op, _ = _trace_to_compute_op(producer_map, inp)
            if compute_op is not None and compute_op.op_type in _LORA_COMPUTE_OPS:
                mul_lora_input = inp

    if mul_lora_input is None or scale_input is None:
        return None

    # lora_B: trace through passthrough ops
    lora_b_matmul, _ = _trace_to_compute_op(producer_map, mul_lora_input)
    if lora_b_matmul is None or lora_b_matmul.op_type not in _LORA_COMPUTE_OPS:
        return None

    # Find lora_B's weight init
    lora_b_init, lora_b_actual_init = _find_weight_init(
        lora_b_matmul, init_names, producer_map
    )
    if lora_b_init is None:
        return None

    # lora_A: trace through passthrough ops from lora_B's non-weight input
    lora_a_matmul = None
    for inp in lora_b_matmul.input:
        if inp == lora_b_init:
            continue
        compute_op, _ = _trace_to_compute_op(producer_map, inp)
        if compute_op is not None and compute_op.op_type in _LORA_COMPUTE_OPS:
            lora_a_matmul = compute_op
            break
    if lora_a_matmul is None:
        return None

    # Find lora_A's weight init
    lora_a_init, _ = _find_weight_init(lora_a_matmul, init_names, producer_map)
    if lora_a_init is None:
        return None

    # Extract module path — for chained Adds, use lora_A node name
    # (the Add node name may just be "Add_1" without module path)
    module_path = _extract_module_path_from_lora_node(lora_a_matmul.name)
    if module_path is None:
        # Fallback: try the Add node name
        module_path = _extract_module_path(add_node.name)
    if module_path is None:
        return None

    target_module = module_path.split(".")[-1]

    # Extract adapter name
    adapter_name = _extract_adapter_name(lora_a_matmul.name, lora_a_init, init_names)

    return _LoRAPattern(
        module_path=module_path,
        target_module=target_module,
        adapter_name=adapter_name,
        add_node=add_node,
        mul_node=mul_node,
        lora_b_matmul=lora_b_matmul,
        lora_a_matmul=lora_a_matmul,
        base_node=None,
        lora_a_init=lora_a_init,
        lora_b_init=lora_b_init,
        lora_b_actual_init=lora_b_actual_init,
        scale_input=scale_input,
        scale_value=scale_value,
        is_chained=True,
        chain_parent_add=chain_parent_add,
    )


def _extract_module_path_from_lora_node(node_name: str) -> str | None:
    """Extract module path from a lora_A/lora_B node name.

    For chained patterns, the Add node name may not contain the full module
    path. Instead we parse the lora_A MatMul node name which encodes it::

        /base_model/q_proj/medical/MatMul → q_proj

    Takes everything before the adapter name segment (which comes after
    the module path segments).
    """
    if not node_name or not node_name.startswith("/"):
        return None

    tokens = node_name.strip("/").split("/")
    if len(tokens) < 3:
        return None

    # Drop trailing op type
    tokens = tokens[:-1]

    # Strip PeftModel wrapper prefixes
    while tokens and tokens[0] in ("model", "base_model"):
        tokens = tokens[1:]

    # Strip PEFT internal names and adapter-related segments
    # The adapter name is the LAST segment (since module path comes first)
    # e.g., ["q_proj", "code"] → module path is everything except last
    tokens = [t for t in tokens if t not in ("base_layer", "lora_A", "lora_B")]

    if not tokens:
        return None

    # Last token is the adapter name — remove it
    # (adapter names don't contain dots, module segments might)
    if len(tokens) > 1:
        tokens = tokens[:-1]

    # Hierarchical deduplication
    deduped = []
    for i, token in enumerate(tokens):
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if "." in next_token and token in next_token:
                continue
        deduped.append(token)

    return ".".join(deduped) if deduped else None


def _find_weight_init(
    matmul_node: onnx.NodeProto,
    init_names: set,
    producer_map: dict,
) -> tuple[str | None, str | None]:
    """Find the weight initializer consumed by a MatMul node.

    Handles both direct consumption and consumption via Identity nodes
    (TorchScript shares zero-initialized tensors via Identity).

    :return: ``(init_name_consumed_by_node, actual_init_name)`` where
        ``actual_init_name`` is the real initializer (before Identity).
        Both are the same when there's no Identity.
    """
    for inp in matmul_node.input:
        # Direct: input is an initializer
        if inp in init_names:
            return inp, inp
        # Indirect: input is from Identity node whose input is an initializer
        prod = producer_map.get(inp)
        if prod is not None and prod.op_type == "Identity":
            identity_input = prod.input[0]
            if identity_input in init_names:
                return inp, identity_input
    return None, None


# ---------------------------------------------------------------------------
# Adapter name extraction
# ---------------------------------------------------------------------------


def _extract_adapter_name(node_name: str, init_name: str, init_names: set) -> str:
    """Extract LoRA adapter name from node name or initializer name.

    Supports two export modes:

    - **dynamo=False**: Adapter name is a path segment in the node name.
      ``/base_model/q_proj/code/MatMul`` → ``"code"``
      For lora_B nodes the segment has a ``_1`` suffix (ModuleDict ordering):
      ``/base_model/q_proj/code_1/MatMul`` → ``"code"``

    - **dynamo=True**: Adapter name is embedded in the init name.
      ``base_model.model.q_proj.lora_A.code.weight`` → ``"code"``

    Falls back to ``"default"`` if no adapter name can be determined.
    """
    # Strategy 1: Parse from init name (dynamo=True produces named inits)
    # Pattern: ...lora_A.{adapter_name}.weight or ...lora_B.{adapter_name}.weight
    if init_name in init_names or "lora_A" in init_name or "lora_B" in init_name:
        parts = init_name.split(".")
        for i, part in enumerate(parts):
            if part in ("lora_A", "lora_B") and i + 2 < len(parts):
                # Next part is adapter name, last part is "weight"
                candidate = parts[i + 1]
                if candidate != "weight":
                    return candidate

    # Strategy 2: Parse from node name (dynamo=False produces structured names)
    # Pattern: /base_model/{module}/{adapter_name}/MatMul
    # TorchScript uses numeric suffixes for repeated modules across layers:
    #   Layer 0: /q_proj/tarot/MatMul (lora_A), /q_proj/tarot_1/MatMul (lora_B)
    #   Layer 1: /q_proj/tarot_2/MatMul (lora_A), /q_proj/tarot_3/MatMul (lora_B)
    # We strip ALL trailing _N suffixes to get the base adapter name.
    if node_name and "/" in node_name:
        tokens = node_name.strip("/").split("/")
        if len(tokens) >= 3:
            # The adapter name is typically the second-to-last segment
            candidate = tokens[-2]
            # Skip known structural segments
            if candidate not in (
                "base_layer",
                "model",
                "base_model",
                "lora_A",
                "lora_B",
            ) and not candidate.startswith("layers"):
                # Strip trailing _N suffix (TorchScript numeric dedup)
                stripped = re.sub(r"_\d+$", "", candidate)
                # Avoid returning module names (standard projection names)
                if "." not in stripped and stripped not in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "self_attn",
                    "mlp",
                ):
                    return stripped

    return "default"


# ---------------------------------------------------------------------------
# Module path extraction
# ---------------------------------------------------------------------------


def _extract_module_path(node_name: str) -> str | None:
    """Extract the PyTorch module path from an ONNX node name.

    ``/model/attn/q_proj/Add`` -> ``attn.q_proj``

    Strips:
    - Leading ``/`` and trailing op type (last token)
    - Known PeftModel wrapper prefixes: ``model``, ``base_model``
    - PEFT internal names: ``base_layer``

    Applies hierarchical deduplication (ported from
    ``aimet_torch/onnx_utils.py:get_pytorch_name_from_onnx_name``).
    """
    if not node_name or not node_name.startswith("/"):
        return None

    tokens = node_name.strip("/").split("/")
    if len(tokens) < 2:
        return None

    # Drop the op type suffix (last token: Add, Mul, MatMul, etc.)
    tokens = tokens[:-1]

    # Strip PeftModel wrapper prefixes
    while tokens and tokens[0] in ("model", "base_model"):
        tokens = tokens[1:]

    # Strip PEFT internal layer names
    tokens = [t for t in tokens if t not in ("base_layer",)]

    if not tokens:
        return None

    # Hierarchical deduplication: when a parent token is a prefix of the
    # child token (e.g., /layer2/layer2.0/), drop the parent.
    deduped = []
    for i, token in enumerate(tokens):
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if "." in next_token and token in next_token:
                continue
        deduped.append(token)

    return ".".join(deduped)


# ---------------------------------------------------------------------------
# Shared initializer splitting
# ---------------------------------------------------------------------------


def _split_shared_initializers(
    model: onnx.ModelProto,
    patterns: list[_LoRAPattern],
) -> None:
    """Split initializers shared via Identity nodes into independent copies.

    ``dynamo=False`` exports share zero-initialized tensors (e.g., lora_B
    weights) via ``Identity`` nodes.  After fine-tuning, each LoRA module
    has different weights, so we need separate initializers.

    For each pattern where ``lora_b_init != lora_b_actual_init`` (i.e.,
    weight comes via Identity), creates a new initializer and updates
    the Identity node's output consumers to use it directly.
    """
    init_map = {init.name: init for init in model.graph.initializer}

    # Group patterns by their actual (pre-Identity) initializer
    shared_groups: dict[str, list[_LoRAPattern]] = {}
    for p in patterns:
        if p.lora_b_init != p.lora_b_actual_init:
            shared_groups.setdefault(p.lora_b_actual_init, []).append(p)

    if not shared_groups:
        return

    identity_nodes_to_remove = set()

    for actual_init_name, group in shared_groups.items():
        source_init = init_map[actual_init_name]
        source_data = numpy_helper.to_array(source_init)

        for pattern in group:
            # Create a new initializer with unique name (use full module_path
            # to avoid collisions across layers, e.g. layers.0.down_proj vs
            # layers.1.down_proj)
            safe_path = pattern.module_path.replace(".", "_")
            new_name = f"{actual_init_name}_split_{safe_path}"
            new_init = numpy_helper.from_array(source_data.copy(), name=new_name)
            model.graph.initializer.append(new_init)
            init_map[new_name] = new_init

            # Find the Identity node that produces lora_b_init
            for node in model.graph.node:
                if node.op_type == "Identity" and node.output[0] == pattern.lora_b_init:
                    identity_nodes_to_remove.add(id(node))
                    break

            # Update the MatMul to consume the new init directly
            for i, inp in enumerate(pattern.lora_b_matmul.input):
                if inp == pattern.lora_b_init:
                    pattern.lora_b_matmul.input[i] = new_name

            # Update the pattern
            pattern.lora_b_init = new_name
            pattern.lora_b_actual_init = new_name

    # Remove Identity nodes that are no longer needed
    if identity_nodes_to_remove:
        # Check ALL nodes' inputs (including candidates) to avoid
        # accidentally removing nodes that non-LoRA ops depend on.
        all_inputs = set()
        for node in model.graph.node:
            all_inputs.update(node.input)

        nodes_to_keep = []
        for node in model.graph.node:
            if id(node) in identity_nodes_to_remove:
                if any(out in all_inputs for out in node.output):
                    nodes_to_keep.append(node)
            else:
                nodes_to_keep.append(node)
        model.graph.ClearField("node")
        model.graph.node.extend(nodes_to_keep)

    logger.info(
        "Split %d shared initializers", sum(len(g) for g in shared_groups.values())
    )


# ---------------------------------------------------------------------------
# Shape restructuring for PEFT compatibility
# ---------------------------------------------------------------------------


def _detect_transposed_params(
    model: onnx.ModelProto,
    patterns: list[_LoRAPattern],
) -> dict[str, tuple[int, ...]]:
    """Detect which LoRA params are stored transposed vs PEFT convention.

    ``dynamo=False`` stores weights in MatMul convention (transposed vs PEFT):
    - ONNX lora_A: ``(in_features, rank)`` — PEFT convention: ``(rank, in_features)``
    - ONNX lora_B: ``(rank, out_features)`` — PEFT convention: ``(out_features, rank)``

    Instead of inserting Transpose nodes (which break AdaScale's onnx2torch
    block conversion), we record the MatMul-convention shape here and
    transpose safetensors weights at load time in ``_remap_safetensors_to_onnx``.

    :return: Dict mapping init name to its MatMul-convention shape.
        Only includes 2D params that differ from PEFT convention.
        Empty dict for ``dynamo=True`` exports (already in PEFT convention).
    """
    init_map = {init.name: init for init in model.graph.initializer}
    transposed = {}

    for pattern in patterns:
        for init_name in [pattern.lora_a_init, pattern.lora_b_init]:
            init = init_map.get(init_name)
            if init is None:
                continue
            shape = tuple(init.dims)
            if len(shape) == 2:
                transposed[init_name] = shape

    logger.info(
        "Detected %d transposed LoRA params (MatMul convention)", len(transposed)
    )
    return transposed


# ---------------------------------------------------------------------------
# Rename and scale conversion
# ---------------------------------------------------------------------------


def _rename_initializers(model: onnx.ModelProto, rename_map: dict[str, str]) -> None:
    """Rename initializers and all references in-place."""
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


def _convert_scale_constants(
    model: onnx.ModelProto,
    patterns: list[_LoRAPattern],
    is_multi_adapter: bool = False,
) -> dict[str, float]:
    """Convert LoRA scale Constant nodes to named initializers.

    PEFT export produces ``alpha/rank`` as anonymous Constant nodes feeding
    the ``Mul`` node.  This replaces each with a named initializer so the
    scale value can be tracked and overridden per adapter at runtime.

    :param is_multi_adapter: If True, include adapter name in scale names.
    :return: Dict mapping scale initializer name to its float value.
    """
    producer_map = {}
    for node in model.graph.node:
        for out in node.output:
            producer_map[out] = node

    init_names = {init.name for init in model.graph.initializer}
    node_by_name = {n.name: n for n in model.graph.node}
    scales = {}

    for pattern in patterns:
        if is_multi_adapter:
            scale_name = (
                f"{_PEFT_PREFIX}{pattern.module_path}.lora_scale.{pattern.adapter_name}"
            )
        else:
            scale_name = f"{_PEFT_PREFIX}{pattern.module_path}.lora_scale"

        # Check if scale is already a named initializer
        if pattern.scale_input in init_names and "lora_scale" in pattern.scale_input:
            continue

        # Get scale value from the Constant node
        scale_value = pattern.scale_value
        if scale_value is None:
            prod = producer_map.get(pattern.scale_input)
            if prod and prod.op_type == "Constant":
                for attr in prod.attribute:
                    if attr.name == "value":
                        scale_value = float(numpy_helper.to_array(attr.t))

        if scale_value is None:
            logger.warning(
                "Could not extract scale value for %s — skipping",
                pattern.module_path,
            )
            continue

        # Create named initializer
        scale_data = np.array(scale_value, dtype=np.float32)
        scale_init = numpy_helper.from_array(scale_data, name=scale_name)
        model.graph.initializer.append(scale_init)

        # Look up live Mul node by name (pattern ref may be stale)
        # Config-driven patterns may have mul_node=None — find by scale_input
        if pattern.mul_node is not None:
            mul_node = node_by_name.get(pattern.mul_node.name)
        else:
            # Find the Mul node that consumes the scale input
            mul_node = None
            for n in model.graph.node:
                if n.op_type == "Mul" and pattern.scale_input in n.input:
                    mul_node = n
                    break

        if mul_node is None:
            logger.warning("Could not find Mul node for %s", pattern.module_path)
            continue

        # Update Mul node input
        for i, inp in enumerate(mul_node.input):
            if inp == pattern.scale_input:
                mul_node.input[i] = scale_name

        scales[scale_name] = scale_value

    # Orphaned scale Constants (no longer consumed after Mul rewiring) are left
    # in the graph. They are harmless — ORT ignores unreferenced nodes.
    # Removing them risks accidentally dropping non-LoRA Constants that share
    # names with LoRA patterns (e.g., causal mask Constants consumed by LessOrEqual).

    logger.info("Converted %d scale constants to named initializers", len(scales))
    return scales


# ---------------------------------------------------------------------------
# Activation tracing (self-contained — avoids quantsim import chain)
# ---------------------------------------------------------------------------


def _trace_lora_activations(
    model: onnx.ModelProto, lora_param_names: list[str]
) -> list[str]:
    """Find activation tensor names in LoRA branches for quantizer placement.

    Traces downstream from LoRA weight initializers through each LoRA branch
    and collects intermediate tensor names.  Stops at the ``Add`` node where
    the LoRA branch merges back into the base path.
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

                all_lora = all(inp in lora_tensors for inp in non_const_inputs)
                if node.op_type == "Add" and not all_lora:
                    continue

                is_quantized = node.op_type not in _OP_OUTPUTS_TO_IGNORE

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

    logger.info("Traced %d LoRA activation tensors", len(activation_names))
    return activation_names


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


def _cleanup_external_data(output_dir: Path, model_filename: str) -> None:
    """Remove existing external data files to avoid conflicts when saving."""
    for data_file in output_dir.glob(f"{model_filename}.data*"):
        data_file.unlink(missing_ok=True)
