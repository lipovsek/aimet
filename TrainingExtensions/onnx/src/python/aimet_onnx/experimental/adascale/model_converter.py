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

filter_op = ["MatMul", "Conv"]


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
    onnx_ir.external_data.load_to_model(subgraph_model)
    param_map = _get_onnx_block_info(subgraph_model)
    return convert(onnx_ir.to_proto(subgraph_model)), param_map


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

        if isinstance(module, torch.nn.Linear):
            pytorch_weight = pytorch_weight.T

        onnx_tensor_name = param_map[name]
        onnx_param_tensor = onnx_model.graph.initializers[onnx_tensor_name]
        if tuple(pytorch_weight.shape) != tuple(onnx_param_tensor.const_value.shape):
            raise ValueError(
                f"pt param shape {pytorch_weight.shape} did not match onnx shape {onnx_param_tensor.const_value.shape}"
            )
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
