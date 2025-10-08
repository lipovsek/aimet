# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any
from packaging.version import parse
import torch
from torch.export import ExportedProgram
import torch.fx.node
from torch.fx.passes.shape_prop import TensorMetadata
from torch._subclasses.fake_tensor import FakeTensorMode
from ..onnx._export import _precompute_encodings
from ...nn import QuantizationMixin


def export(mod: torch.nn.Module, *args, **kwargs) -> ExportedProgram:
    """
    Export :class:`QuantizationSimModel` to ExportedProgram with
    quantization ops embedded in the aten graph.

    This function takes set of same arguments as `torch.export.export()`_
    """
    # pylint: disable=protected-access
    if parse(torch.__version__) < parse("2.8.0"):
        raise RuntimeError(
            "Exporting to torch.exoprt.ExportedProgram is only supported with torch>=2.8.0; "
            f" got torch=={torch.__version__}"
        )

    untraceable_modules = {
        name: module
        for name, module in mod.named_modules()
        if isinstance(module, QuantizationMixin) and not module._is_dynamo_traceable()
    }

    if untraceable_modules:
        raise RuntimeError(
            "Following modules don't support dynamo tracing:\n"
            + "\n".join(
                [
                    f"- {name} (type: {type(module).__name__})"
                    for name, module in untraceable_modules.items()
                ]
            )
        )

    # Pre-compute scale and offset to omit verbose
    # scale/offset derivation logic in the exported graph
    with _precompute_encodings(mod), torch.no_grad():
        ep = torch.export.export(mod, *args, **kwargs)

    for q_dq_node in ep.graph.nodes:
        if not (
            q_dq_node.op == "call_function"
            and isinstance(q_dq_node.target, torch._ops.OpOverload)
        ):
            continue
        if (
            q_dq_node.target.name().startswith("aten::fake_quantize")
            or q_dq_node.target.name().startswith("quantized_decomposed::quantize")
            or q_dq_node.target.name().startswith("quantized_decomposed::dequantize")
        ):
            _fold_scale_and_zp(q_dq_node, ep)

    _remove_dangling_nodes(ep)
    return ep


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

    # Clean up graph_signature and state_dict
    ep.graph_signature.input_specs = [
        input_spec
        for input_spec in ep.graph_signature.input_specs
        if ep.graph.find_nodes(op="placeholder", target=input_spec.arg.name, sort=False)
    ]
    all_targets: set[str | None] = set(
        input_spec.target for input_spec in ep.graph_signature.input_specs
    )

    for dangling_key in ep.state_dict.keys() - all_targets:
        del ep.state_dict[dangling_key]

    for dangling_key in ep.constants.keys() - all_targets:
        del ep.constants[dangling_key]


def _fold_scale_and_zp(q_dq_node: torch.fx.Node, ep: ExportedProgram):
    if len(q_dq_node.all_input_nodes) > 1:
        scale: torch.Tensor = _eval_node(q_dq_node.all_input_nodes[1], ep)
        scale_placeholder: torch.fx.Node = _insert_placeholder(
            ep,
            val=scale,
            node_name=f"p_{q_dq_node.name}_scale",
            tensor_name=f"{q_dq_node.name}_scale",
            consumer=q_dq_node,
        )
        q_dq_node.replace_input_with(q_dq_node.all_input_nodes[1], scale_placeholder)

    if len(q_dq_node.all_input_nodes) > 2:
        zero_point: torch.Tensor = _eval_node(q_dq_node.all_input_nodes[2], ep)
        zero_point_placeholder: torch.fx.Node = _insert_placeholder(
            ep,
            val=zero_point,
            node_name=f"p_{q_dq_node.name}_zero_point",
            tensor_name=f"{q_dq_node.name}_zero_point",
            consumer=q_dq_node,
        )
        q_dq_node.replace_input_with(
            q_dq_node.all_input_nodes[2], zero_point_placeholder
        )


def _insert_placeholder(
    ep: ExportedProgram,
    val: torch.Tensor,
    node_name: str,
    tensor_name: str,
    consumer: torch.fx.Node,
):
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

    with ep.graph.inserting_before(consumer):
        node = ep.graph.create_node(
            op="placeholder",
            target=node_name,
            name=node_name,
        )
    fake_mode = FakeTensorMode()
    converter = fake_mode.fake_tensor_converter
    fake_tensor = converter.from_real_tensor(fake_mode, val)
    node.meta.update(
        {
            "val": fake_tensor,
            "example_value": fake_tensor,
            "tensor_metadata": TensorMetadata(
                shape=val.shape,
                dtype=val.dtype,
                requires_grad=val.requires_grad,
                stride=val.stride(),
                memory_format=torch.contiguous_format,
                is_quantized=False,
                qparams={},
            ),
            "seq_nr": 1,
            # "from_node": [],
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
