# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.experimental.adascale.quantizer import QuantizedLinear, QuantizedConv2d
from aimet_onnx.qc_quantize_op import QcQuantizeOp

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)
import onnx_ir
from onnx.utils import Extractor
from onnx2torch import convert
from aimet_onnx.experimental.adascale.onnx2torch_ext import *  # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_onnx import ir_utils
from onnx2torch.onnx_graph import OnnxGraph
from typing import Tuple, List, Dict, Collection
from aimet_onnx.common.quantsim import calculate_delta_offset

filter_op = ["MatMul", "Conv", "Gemm"]


def _get_onnx_subgraph(
    extractor: Extractor,
    block_input_output_names: Tuple[List[str], List[str]],
):
    """
    Given a onnx block end points get onnx subgraph
    """
    block_input_names, block_output_names = block_input_output_names
    try:
        block_fp32_model = extractor.extract_model(
            block_input_names,
            block_output_names,
        )
        return block_fp32_model
    except Exception:
        raise RuntimeError(  # pylint: disable=raise-missing-from
            f"Unable to extract onnx subgraph for given block input/output {block_input_output_names}"
        )


def _get_onnx_block_info(onnx_subgraph: onnx_ir.Model):
    """
    For an onnx subgraph get onnx param name from initializer list map
    """
    graph = onnx_subgraph.graph
    name_to_node_filtered = {
        n.name: n for n in graph.all_nodes() if n.op_type in filter_op
    }
    node_name_to_onnx_param = {}
    for node in name_to_node_filtered.values():
        # TODO remove using "bias" word search and add op specific logic instead
        if node.op_type == "Conv":
            node_name_to_onnx_param[OnnxGraph.generate_node_name(node)] = node.inputs[
                1
            ].name
        else:
            for edge in node.inputs:
                if (
                    edge.name in onnx_subgraph.graph.initializers
                    and "bias" not in edge.name
                ):
                    # Bias will not be updated so we donot need to keep track of bias
                    node_name_to_onnx_param[OnnxGraph.generate_node_name(node)] = (
                        edge.name
                    )
    return node_name_to_onnx_param


def resolve_block_residual_name(graph: onnx_ir.Graph, value_name: str) -> str:
    """
    Walk back through leading ``Cast`` producers and return the deepest
    pre-Cast value name.
    Used to recover the true cross-block residual when the RMSNorm anchor's input is post-Cast (fp16 graphs).
    """
    name_to_value = onnx_ir.convenience.create_value_mapping(graph)
    if value_name not in name_to_value:
        return value_name
    value = name_to_value[value_name]
    while True:
        producer = value.producer()
        if producer is None or producer.op_type != "Cast":
            break
        upstream = producer.inputs[0]
        if upstream is None or upstream.name is None:
            break
        value = upstream
    return value.name


def required_extra_block_inputs(
    graph: onnx_ir.Graph,
    input_names: List[str],
    output_names: List[str],
) -> List[str]:
    """
    Return graph-input names the subgraph requires beyond ``input_names``.

    Walks back from ``output_names`` with ``input_names`` as a barrier; any
    producer-less, non-initializer value reached is an unbounded graph input.
    """

    name_to_value = onnx_ir.convenience.create_value_mapping(graph)
    declared = {name_to_value[n] for n in input_names if n in name_to_value}
    visited_values = set(declared)
    visited_nodes = set()
    stack = [name_to_value[n] for n in output_names if n in name_to_value]
    extras: List[str] = []
    seen = set(input_names)

    while stack:
        value = stack.pop()
        if value in visited_values:
            continue
        visited_values.add(value)
        producer = value.producer()
        if producer is None:
            if not value.is_initializer() and value.name and value.name not in seen:
                extras.append(value.name)
                seen.add(value.name)
            continue
        if producer in visited_nodes:
            continue
        visited_nodes.add(producer)
        for inp in producer.inputs:
            if inp is None or inp in visited_values:
                continue
            stack.append(inp)

    # Normalize order to graph input order
    return [inp.name for inp in graph.inputs if inp.name in extras]


def get_pt_block(
    model: onnx_ir.Model, block_input_output_names: Tuple[List[str], List[str]]
):
    """
    Given a onnx block end points get a pytorch block
    :param model: onnx.ModelProto
    :param block_input_output_names: input/output names for block end points
    """
    input_names, output_names = block_input_output_names

    subgraph = onnx_ir.convenience.extract(
        model.graph,
        input_names,
        output_names,
    )
    subgraph_model = onnx_ir.Model(
        subgraph, ir_version=model.ir_version, functions=list(model.functions.values())
    )
    ir_utils.remove_aimet_quantizers(subgraph_model)
    ir_utils.inline_all_supergroups(subgraph_model)
    onnx_ir.passes.common.TopologicalSortPass().call(subgraph_model)
    onnx_ir.external_data.load_to_model(subgraph_model)
    param_map = _get_onnx_block_info(subgraph_model)
    return convert(onnx_ir.to_proto(subgraph_model)), param_map


def _get_tensor_consumers(tensor: onnx_ir.Value):
    consumers = set()
    for consumer, _ in tensor.uses():
        if consumer.op_type in ("Identity", "QcQuantizeOp"):
            consumers.update(_get_tensor_consumers(consumer.outputs[0]))
            continue
        consumers.add(consumer)
    return consumers


def _should_transpose_weight(module: torch.nn.Module, weight: onnx_ir.Value):
    if not isinstance(module, torch.nn.Linear):
        return False

    def _is_transposed_weight(node: onnx_ir.Node):
        if node.op_type not in ("MatMul", "Gemm"):
            return False
        if node.op_type == "MatMul":
            return True

        trans_b = node.attributes.get("transB", 0)
        if trans_b:
            trans_b = trans_b.as_int()

        return not trans_b

    consumers = _get_tensor_consumers(weight)

    if not any(_is_transposed_weight(node) for node in consumers):
        return False

    if not all(_is_transposed_weight(node) for node in consumers):
        raise RuntimeError(f"Conflicting uses of {weight} by consumers {consumers}")

    return True


def copy_pt_weights_to_onnx(
    pt_block: torch.fx.GraphModule,
    onnx_model: onnx_ir.Model,
    param_map: Collection[Dict[str, str]],
    quantizer_dict: Dict[str, QcQuantizeOp] = None,
):
    """
    Given a pt_block with adascale params computed, copy the params to onnx model
    :param pt_block: pytorch block with adascale weight quantizers
    :param onnx_model: onnx model before adascale
    :param pt_weights_to_onnx_initializers: Mapping between PT weight names to ONNX initializers
    :param quantizer_dict: Optional quantizer dict; params whose quantizer is
        disabled are skipped (e.g. LoRA params during base-model AdaScale).
    """
    for name, module in pt_block.named_modules():
        if param_map.get(name) is None:
            continue
        if quantizer_dict is not None and not quantizer_dict[param_map[name]].enabled:
            continue
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            pytorch_weight = (
                module.param_quantizers["weight"]
                .get_folded_weight(module.weight)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            pytorch_weight = module.weight.detach().cpu().numpy()

        onnx_tensor_name = param_map[name]
        onnx_param_tensor = onnx_model.graph.initializers[onnx_tensor_name]
        if _should_transpose_weight(module, onnx_param_tensor):
            pytorch_weight = pytorch_weight.T
        if tuple(pytorch_weight.shape) != tuple(onnx_param_tensor.const_value.shape):
            raise ValueError(
                f"pt param shape {pytorch_weight.shape} did not match onnx shape {onnx_param_tensor.const_value.shape}"
            )
        # Preserve the original initializer dtype so downstream ONNX consumers
        # (e.g. the activation sampler's ORT session) keep type-consistent
        # MatMul inputs.
        onnx_dtype = onnx_param_tensor.const_value.dtype.numpy()
        if pytorch_weight.dtype != onnx_dtype:
            pytorch_weight = pytorch_weight.astype(onnx_dtype)
        onnx_param_tensor.const_value = onnx_ir.Tensor(pytorch_weight)
        _logger.info(
            "Copy from PyTorch to ONNX: torch : %s  onnx param : %s",
            name,
            onnx_tensor_name,
        )


def copy_pt_encodings_to_sim(
    pt_block: torch.fx.GraphModule,
    quantizer_dict: Dict[str, QcQuantizeOp],
    pt_weights_to_onnx_initializers: Collection[Dict[str, str]],
):
    """
    Given the PT block with adascale params computed, copy the encodings to sim
    :param pt_block: pytorch block with adascale weight quantizers
    :param quantizer_dict: Dictionary of quantizers
    :param pt_weights_to_onnx_initializers: Mapping between PT weight names to ONNX initializers
    """
    for name, module in pt_block.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            onnx_param_name = pt_weights_to_onnx_initializers[name]
            #### TODO Check the modules
            # copy encodings over to onnx quantizers
            new_min = module.param_quantizers["weight"].get_min().detach().cpu().numpy()
            new_max = module.param_quantizers["weight"].get_max().detach().cpu().numpy()

            enc = quantizer_dict[onnx_param_name].get_encodings()
            if enc is None:
                # quantizer is disabled (e.g. LoRA params skipped during AdaScale) — skip
                continue
            if len(new_min) != len(enc) or len(new_max) != len(enc):
                raise RuntimeError(
                    "Encodings of the onnx quantizer and adascale quantizer have different lengths"
                )

            for i, encoding in enumerate(enc):
                delta, offset = calculate_delta_offset(
                    min_val=new_min[i],
                    max_val=new_max[i],
                    bitwidth=module.param_quantizers["weight"].bitwidth,
                    use_symmetric_encodings=True,
                    use_strict_symmetric=False,
                )
                # TODO: #6393 calculate_delta_offset to return float
                encoding.delta = delta.item()
                encoding.offset = offset.item()
                encoding.min = new_min[i].item()
                encoding.max = new_max[i].item()
            quantizer_dict[onnx_param_name].load_encodings(enc)
            quantizer_dict[onnx_param_name].freeze_encodings()
