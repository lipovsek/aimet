# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access
from typing import Any
import operator

import torch
import torch.fx.node
from torch.export import ExportedProgram
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._subclasses.fake_tensor import FakeTensorMode


def _is_grid_preserving_op(node: torch.fx.Node) -> bool:
    if node.target is operator.getitem:
        return True

    if not isinstance(node.target, torch._ops.OpOverload):
        return False

    name, *_ = node.target.name().split(".")
    return name in (
        "aten::alias",
        "aten::alias_copy",
        "aten::clone",
        "aten::contiguous",
        "aten::copy",
        "aten::copy_",
        "aten::detach",
        "aten::diag",
        "aten::diag_embed",
        "aten::diagonal",
        "aten::diagonal_backward",
        "aten::diagonal_copy",
        "aten::dropout",
        "aten::dropout_",
        "aten::embedding",
        "aten::expand",
        "aten::expand_copy",
        "aten::flatten",
        "aten::gather",
        "aten::item",
        "aten::kthvalue",
        "aten::masked_select",
        "aten::max_pool1d",
        "aten::max_pool2d",
        "aten::max_pool2d_with_indices",
        "aten::max_pool3d",
        "aten::max_pool3d_with_indices",
        "aten::max",
        "aten::min",
        "aten::narrow",
        "aten::narrow_copy",
        "aten::native_dropout",
        "aten::nonzero",
        "aten::pad",
        "aten::permute",
        "aten::permute_copy",
        "aten::reflection_pad1d",
        "aten::reflection_pad2d",
        "aten::reflection_pad3d",
        "aten::relu",
        "aten::relu_",
        "aten::repeat",
        "aten::repeat_interleave",
        "aten::replication_pad1d",
        "aten::replication_pad2d",
        "aten::replication_pad3d",
        "aten::reshape",
        "aten::_reshape_alias",
        "aten::_reshape_copy",
        "aten::rot90",
        "aten::select",
        "aten::slice",
        "aten::squeeze",
        "aten::squeeze_copy",
        "aten::t",
        "aten::t_",
        "aten::t_copy",
        "aten::take",
        "aten::tile",
        "aten::topk",
        "aten::transpose",
        "aten::transpose_",
        "aten::transpose_copy",
        "aten::unfold",
        "aten::unfold_backward",
        "aten::unfold_copy",
        "aten::unsqueeze",
        "aten::unsqueeze_",
        "aten::unsqueeze_copy",
        "aten::view",
        "aten::view_copy",
        "aten::zeros_like",
    )


def _remove_dangling_nodes(ep: ExportedProgram):
    output_node = ep.graph.output_node()
    visited: set[torch.fx.Node] = set()
    stack = [output_node]

    # Reverse-DFS from output node
    while stack:
        node = stack.pop(-1)
        if node in visited:
            continue
        visited.add(node)
        stack += node.all_input_nodes

    # Mark all visited nodes as non-dangling node
    dangling_nodes = set(ep.graph.nodes) - visited

    # Remove dangling nodes from graph
    for node in reversed(list(ep.graph.nodes)):
        if node in dangling_nodes:
            ep.graph.erase_node(node)

    ep.graph.eliminate_dead_code()
    ep.graph_module.recompile()

    node_name_to_input_spec = {
        spec.arg.name: spec for spec in ep.graph_signature.input_specs
    }
    # Clean up graph_signature and state_dict
    ep.graph_signature.input_specs = [
        node_name_to_input_spec[input_node.name]
        for input_node in ep.graph.find_nodes(op="placeholder", sort=False)
    ]
    all_targets: set[str | None] = set(
        input_spec.target for input_spec in ep.graph_signature.input_specs
    )

    for dangling_key in ep.state_dict.keys() - all_targets:
        del ep.state_dict[dangling_key]

    for dangling_key in ep.constants.keys() - all_targets:
        del ep.constants[dangling_key]


def _insert_placeholder(
    ep: ExportedProgram,
    val: torch.Tensor,
    node_name: str,
    tensor_name: str,
    consumer: torch.fx.Node,
):
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument
    from torch._export.utils import _detect_fake_mode_from_gm

    with ep.graph.inserting_before(consumer):
        node = ep.graph.create_node(
            op="placeholder",
            target=node_name,
            name=node_name,
        )

    fake_mode = _detect_fake_mode_from_gm(ep.graph_module) or FakeTensorMode()
    converter = fake_mode.fake_tensor_converter
    fake_tensor = converter.from_real_tensor(fake_mode, val)
    node.meta.update(
        {
            "val": fake_tensor,
            "tensor_meta": _extract_tensor_metadata(fake_tensor),
        }
    )

    i = InputSpec(
        kind=InputKind.BUFFER,
        arg=TensorArgument(name=node_name),
        target=tensor_name,
        persistent=True,
    )
    ep.graph_signature.input_specs.append(i)
    ep.state_dict.update({tensor_name: val})

    return node


def _eval_node(
    arg: torch.fx.node.Argument,
    ep: ExportedProgram,
) -> Any:
    input_specs = {spec.arg.name: spec for spec in ep.graph_signature.input_specs}
    params_and_constants = ep.state_dict | ep.constants

    def _do_eval(arg: torch.fx.node.Argument):
        if not isinstance(arg, torch.fx.Node):
            return arg

        node = arg

        if node.op == "placeholder":
            input_spec = input_specs[node.name]
            param_or_const_name = input_spec.target
            if param_or_const_name not in params_and_constants:
                raise RuntimeError(
                    "Couldn't find parameter, buffer, or constant "
                    f"with name {param_or_const_name} of node {node.name}"
                )
            return params_and_constants[param_or_const_name]

        if not callable(node.target):
            raise RuntimeError(
                f"Internal error occurred. Expected node {node.name} (op: {node.op}) "
                f"to be callable, but got node.target of type {type(node.target)}"
            )

        args = tuple(_do_eval(arg) for arg in node.args)
        kwargs = {key: _do_eval(val) for key, val in node.kwargs.items()}

        return node.target(*args, **kwargs)

    return _do_eval(arg)
