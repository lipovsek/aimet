# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-ir related utility functions"""

import onnx_ir
import numpy as np

from .fusion_registry import AIMET_SUPERGROUP_DOMAIN


def get_constant_singleton_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> float | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    numpy_value = get_constant_or_attribute_value(value)

    if numpy_value is None or numpy_value.size != 1:
        return None

    return numpy_value.flatten()[0].item()


def get_constant_as_array(value: onnx_ir.Value | None) -> np.ndarray | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    if value is None:
        return None

    const_value = onnx_ir.convenience.get_const_tensor(value)
    if const_value is None:
        return None

    return const_value.numpy()


def get_constant_or_attribute_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> None | np.ndarray:
    """Get the constant value from an ONNX IR Value or Attr, if it exists."""
    if value is None:
        return None
    if isinstance(value, onnx_ir.Value):
        return get_constant_as_array(value)
    if isinstance(value, onnx_ir.Attr):
        return np.asarray(value.value)
    raise RuntimeError(f"Received unexpected type for value: {type(value)}")


def _sort_functions_hierarchically(model: onnx_ir.Model) -> None:
    """Sort model functions from outermost to innermost to prevent mangling of names during inlining."""
    # pylint: disable=protected-access
    sorted_funcs = {}

    def node_has_impl(node: onnx_ir.Node) -> bool:
        return (
            node.domain != AIMET_SUPERGROUP_DOMAIN
            or node.op_identifier() in sorted_funcs
        )

    while True:
        runnable_functions = {
            fid: func
            for fid, func in model.functions.items()
            if fid not in sorted_funcs
            and all(node_has_impl(n) for n in func.graph.all_nodes())
        }
        if not runnable_functions:
            break
        sorted_funcs.update(runnable_functions)

    if not sorted_funcs.keys() == model.functions.keys():
        raise RuntimeError(
            f"Cycle detected among supergroup functions: {set(model.functions.keys()) - set(sorted_funcs.keys())}"
        )

    # Reverse ordering to prevent name mangling while unrolling
    model._functions = dict(reversed(list(sorted_funcs.items())))


def inline_all_supergroups(model: onnx_ir.Model) -> None:
    """Inline all aimet supergroup functions, restoring original node and value names."""
    supergroup_functions = {
        func
        for func in model.functions.values()
        if func.domain == AIMET_SUPERGROUP_DOMAIN
    }
    if not supergroup_functions:
        return

    _sort_functions_hierarchically(model)
    onnx_ir.passes.common.InlinePass(lambda f: f in supergroup_functions).call(model)


def unique_name(base: str, existing: set[str]) -> str:
    """Generate a unique name based on the provided base that does not exist in the existing set."""
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def get_upstream_cast_type(value: onnx_ir.Value) -> int | None:
    """Return the ``to`` attribute of an upstream Cast producer, if any"""
    producer = value.producer()
    if producer is None or producer.op_type != "Cast" or producer.domain != "":
        return None
    to_attr = producer.attributes.get("to")
    return to_attr.as_int() if to_attr is not None else None
