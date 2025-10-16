# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access
from typing import Any, Tuple, Optional
from packaging.version import parse
import torch
from torch.export import ExportedProgram
import torch.fx.node
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._subclasses.fake_tensor import FakeTensorMode
from ..onnx._export import _precompute_encodings
from ...nn import QuantizationMixin
from ...quantization.affine import AffineQuantizerBase


def export(mod: torch.nn.Module, *args, **kwargs) -> ExportedProgram:
    """
    Export :class:`QuantizationSimModel` to ExportedProgram with
    quantization ops embedded in the aten graph.

    This function takes set of same arguments as `torch.export.export()`_
    """
    if parse(torch.__version__) < parse("2.8.0"):
        raise RuntimeError(
            "Exporting to torch.exoprt.ExportedProgram is only supported with torch>=2.8.0; "
            f" got torch=={torch.__version__}"
        )

    #  If no quantizers are initialized, raise error
    if all(
        not qtzr.is_initialized()
        for qtzr in mod.modules()
        if isinstance(qtzr, AffineQuantizerBase)
    ):
        raise RuntimeError(
            "Please ensure that the quantizers are initialized before exporting. "
            "You can do this by running a forward pass with representative data "
            "within QuantizationSimModel.compute_encodings() or "
            "under aimet_torch.nn.compute_encodings() context manager."
        )

    # If any qmodule is not dynamo traceable, raise error
    untraceable_modules = []

    for name, module in mod.named_modules():
        if not isinstance(module, QuantizationMixin):
            continue
        is_dynamo_traceable, reason = module._is_dynamo_traceable()
        if not is_dynamo_traceable:
            untraceable_modules.append((name, module, reason))

    if untraceable_modules:
        raise RuntimeError(
            "Following modules don't support dynamo tracing:\n"
            + "\n".join(
                [
                    f"- {name} (type: {type(module).__name__}): {reason}"
                    for name, module, reason in untraceable_modules
                ]
            )
        )

    # Pre-compute scale and offset to omit verbose
    # scale/offset derivation logic in the exported graph
    with _precompute_encodings(mod), torch.no_grad():
        ep = torch.export.export(mod, *args, **kwargs)

    for node in ep.graph.nodes:
        if _is_qdq_op(node):
            _try_fold_scale_and_zp(node, ep)

    for node in ep.graph.nodes:
        if _is_grid_preserving_op(node):
            _try_insert_output_qdq(ep, node)

    for node in reversed(ep.graph.nodes):
        if _is_grid_preserving_op(node):
            _try_insert_input_qdq(ep, node)

    _remove_dangling_nodes(ep)
    return ep


def _try_insert_output_qdq(ep: ExportedProgram, node: torch.fx.Node):
    input_q, input_dq = _get_input_qdq(node)
    output_q, output_dq = _get_output_qdq(node)

    if not (
        input_q is not None
        and input_dq is not None
        and output_q is None
        and output_dq is None
    ):
        return

    qtype = input_q.args[5] if len(input_q.args) > 5 else input_q.kwargs["dtype"]
    with ep.graph.inserting_after(node):
        output_q = ep.graph.create_node(
            op=input_q.op,
            target=input_q.target,
            args=(node, *input_q.args[1:]),
            kwargs=input_q.kwargs.copy(),
            name=f"{input_q.name}_copy",
        )
        output_q.meta["val"] = node.meta["val"].to(qtype)
        output_q.meta["tensor_meta"] = _extract_tensor_metadata(output_q.meta["val"])

    with ep.graph.inserting_after(output_q):
        output_dq = ep.graph.create_node(
            op=input_dq.op,
            target=input_dq.target,
            args=(output_q, *input_dq.args[1:]),
            kwargs=input_dq.kwargs.copy(),
            name=f"{input_dq.name}_copy",
        )
        output_dq.meta.update(
            {
                "val": node.meta["val"].clone(),
                "tensor_meta": node.meta["tensor_meta"],
            }
        )

    node.replace_all_uses_with(output_dq)
    output_q.args = (node, *input_q.args[1:])

    ep.graph.eliminate_dead_code()
    ep.graph_module.recompile()


def _try_insert_input_qdq(ep: ExportedProgram, node: torch.fx.Node):
    if node.all_input_nodes:
        input = node.all_input_nodes[0]  # pylint: disable=redefined-builtin
    else:
        return

    input_q, input_dq = _get_input_qdq(node)
    output_q, output_dq = _get_output_qdq(node)

    if not (
        input
        and len(input.users) == 1
        and output_q is not None
        and output_dq is not None
        and input_q is None
        and input_dq is None
    ):
        return

    qtype = output_q.args[5] if len(output_q.args) > 5 else output_q.kwargs["dtype"]
    with ep.graph.inserting_after(input):
        input_q = ep.graph.create_node(
            op=output_q.op,
            target=output_q.target,
            args=(input, *output_q.args[1:]),
            kwargs=output_q.kwargs.copy(),
            name=f"{output_q.name}_copy",
        )
        input_q.meta["val"] = input.meta["val"].to(qtype)
        input_q.meta["tensor_meta"] = _extract_tensor_metadata(input_q.meta["val"])

    with ep.graph.inserting_after(input_q):
        input_dq = ep.graph.create_node(
            op=output_dq.op,
            target=output_dq.target,
            args=(input_q, *output_dq.args[1:]),
            kwargs=output_dq.kwargs.copy(),
            name=f"{output_dq.name}_copy",
        )
        input_dq.meta.update(
            {
                "val": input.meta["val"].clone(),
                "tensor_meta": node.meta["tensor_meta"],
            }
        )

    input.replace_all_uses_with(input_dq)
    input_q.args = (input, *input_q.args[1:])

    ep.graph.eliminate_dead_code()
    ep.graph_module.recompile()


def _get_output_qdq(
    node: torch.fx.Node,
) -> Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node]]:
    (q,) = node.users if len(node.users) == 1 else (None,)
    (dq,) = q.users if q and len(q.users) == 1 else (None,)

    if not (
        dq
        and isinstance(dq.target, torch._ops.OpOverload)
        and dq.target.name().startswith("quantized_decomposed::dequantize_per_tensor")
    ):
        dq = None
        q = None

    if not (
        q
        and isinstance(q.target, torch._ops.OpOverload)
        and q.target.name().startswith("quantized_decomposed::quantize_per_tensor")
    ):
        q = None

    return q, dq


def _get_input_qdq(
    node: torch.fx.Node,
) -> Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node]]:
    dq = node.all_input_nodes[0] if node.all_input_nodes else None
    q = dq.all_input_nodes[0] if dq and dq.all_input_nodes else None

    if not (
        dq
        and isinstance(dq.target, torch._ops.OpOverload)
        and dq.target.name().startswith("quantized_decomposed::dequantize_per_tensor")
    ):
        dq = None
        q = None

    if not (
        q
        and isinstance(q.target, torch._ops.OpOverload)
        and q.target.name().startswith("quantized_decomposed::quantize_per_tensor")
    ):
        q = None

    return q, dq


def _is_qdq_op(node: torch.fx.Node) -> bool:
    if not isinstance(node.target, torch._ops.OpOverload):
        return False

    return (
        node.target.name().startswith("aten::fake_quantize")
        or node.target.name().startswith("quantized_decomposed::quantize")
        or node.target.name().startswith("quantized_decomposed::dequantize")
    )


def _is_grid_preserving_op(node: torch.fx.Node) -> bool:
    if not isinstance(node.target, torch._ops.OpOverload):
        return False

    name, *_ = node.target.name().split(".")
    return name in (
        "aten::dropout",
        "aten::dropout_",
        "aten::expand",
        "aten::gather",
        "aten::flatten",
        "aten::max_pool1d",
        "aten::max_pool2d",
        "aten::max_pool3d",
        "aten::max",
        "aten::min",
        "aten::nonzero",
        "aten::pad",
        "aten::permute",
        "aten::relu",
        "aten::relu_",
        "aten::repeat",
        "aten::repeat_interleave",
        "aten::reshape",
        "aten::slice",
        "aten::squeeze",
        "aten::tile",
        "aten::topk",
        "aten::transpose",
        "aten::unsqueeze",
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


def _try_fold_scale_and_zp(q_dq_node: torch.fx.Node, ep: ExportedProgram):
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
