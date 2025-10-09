# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from onnx import numpy_helper
import torch
from torch.nn import Parameter
import copy

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from aimet_onnx.experimental.adascale.quantizer import QuantizedLinear

from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)
from aimet_onnx.quantsim import QuantizationSimModel
from typing import Dict
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
)

decoder_block_to_layername_map = {
    LlamaDecoderLayer: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ],
    Qwen2DecoderLayer: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ],
}


class ModelConverter:
    """
    Given a quantsim.onnx model and transformers model type:
    1. Get a onnx decoder blocks
    2. Create a pytorch decoder block of model_type
    3. Copy the wts from onnx decoder block to pytorch
    """

    def __init__(self, quantsim: QuantizationSimModel, model_config: Dict):
        self.qsim = quantsim
        self.model_config = model_config
        self._get_onnx_blocks()
        self.pt_block = self.get_pt_decoder_block()
        self.mapping_pt_to_onnx_params = {}
        self.needs_transpose = set()

    def get_pt_decoder_block(self):
        decoder_block_cls = self.model_config.block_type
        config = self.model_config.model_config
        return decoder_block_cls(config, 0)

    def _get_onnx_blocks(self):
        self.end_points = get_decoder_blocks_end_points(self.qsim)
        self.num_layers = len(self.end_points)
        graph = self.qsim.model.model.graph
        self.name_to_node = {n.name: n for n in graph.node}
        self.node_edge_to_parent = {n.output[0]: n.name for n in graph.node}
        self.initializer_name_to_index_map = {
            init.name: idx
            for idx, init in enumerate(self.qsim.model.model.graph.initializer)
        }
        self.all_ops = self.qsim.connected_graph.ordered_ops
        self.op_name_to_index = {
            op.name: index for index, op in enumerate(self.all_ops)
        }

    def get_pt_block(self, blk_idx: int):
        """
        Given a block index:
            1. get the onnx decoder block using onnx block boundaries
            2. get pt_decoder_block[blk_idx]
            3. Copy the wts and bias from onnx decoder block to pytorch decoder block
        Assumptions:
            1. wts are going to be present in initializers list or constant nodes
            2. for self attn block (q,k,v,o_proj matmul is followed by an add. matmul has the wts and add has the bias)
            3. climb up matmuls (only of type Wx determined by name) to get wts
            4. climp up 2 steps "MatMul", "Add", "Mul" op to check for wts or bias
        """
        pt_block = copy.deepcopy(self.pt_block)
        layers_of_interest = decoder_block_to_layername_map[
            self.model_config.block_type
        ]
        all_ops_per_decoder_blk = self.all_ops[
            self.op_name_to_index[
                str(self.end_points[blk_idx][0])
            ] : self.op_name_to_index[str(self.end_points[blk_idx][1])] + 1
        ]

        # to reduce number of loops, filter out onnx ops we will never need
        all_ops_filtered = [
            op for op in all_ops_per_decoder_blk if op.type in ("MatMul", "Add", "Mul")
        ]

        for name, module in pt_block.named_modules():
            layer_type = [(name, elem) for elem in layers_of_interest if elem in name]
            if len(layer_type) == 0:
                continue
            _, elem = layer_type[0]

            target_ops = [
                target_op for target_op in all_ops_filtered if elem in target_op.name
            ]
            if len(target_ops) < 1 or len(target_ops) > 4:
                raise RuntimeError(
                    f"We expect between 1 to 4 onnx nodes whose name contains {elem} but got {len(target_ops)}"
                )
            for target_op in target_ops:
                for edge in self.name_to_node[target_op.name].input:
                    found_param_name = self._get_wt_bias_param(edge)
                    if found_param_name:
                        onnx_initializer_list_idx = self.initializer_name_to_index_map[
                            found_param_name
                        ]
                        onnx_wt_or_bias = numpy_helper.to_array(
                            self.qsim.model.model.graph.initializer[
                                onnx_initializer_list_idx
                            ]
                        )
                        onnx_wt_or_bias_shape = onnx_wt_or_bias.shape
                        rank = len(onnx_wt_or_bias_shape)
                        torch_param = copy.deepcopy(
                            Parameter(
                                torch.from_numpy(onnx_wt_or_bias).to(torch.float32)
                            )
                        )
                        if rank == 2:  # its matmul weight
                            pt_wt_shp = module.weight.shape
                            onnx_wt_or_bias = onnx_wt_or_bias.T
                            onnx_wt_or_bias_shape = onnx_wt_or_bias.shape
                            torch_param = copy.deepcopy(
                                Parameter(
                                    torch.from_numpy(onnx_wt_or_bias).to(torch.float32)
                                )
                            )
                            self.needs_transpose.add(name)
                            if pt_wt_shp != onnx_wt_or_bias_shape:
                                raise RuntimeError(
                                    f"pt wt shape {pt_wt_shp} did not match onnx shape {onnx_wt_or_bias_shape} (with ot without transpose)"
                                )
                            module.weight = torch_param
                            self.mapping_pt_to_onnx_params[name] = {
                                "weight": found_param_name
                            }
                            _logger.info(
                                "Copy from Onnx to PyTorch: torch : %s weight from onnx param : %s",
                                name,
                                found_param_name,
                            )
                        elif rank == 1:  # its matmul bias or layernorm wt
                            is_matmul_bias = hasattr(
                                module, "bias"
                            )  # layernorm doesnt have "bias"

                            if name not in self.mapping_pt_to_onnx_params:
                                self.mapping_pt_to_onnx_params[name] = {}

                            if is_matmul_bias:
                                if module.bias.shape != onnx_wt_or_bias_shape:
                                    raise RuntimeError(
                                        f"pt bias shape {module.bias.shape} did not match onnx shape {onnx_wt_or_bias_shape}"
                                    )

                                module.bias = torch_param
                                self.mapping_pt_to_onnx_params[name]["bias"] = (
                                    found_param_name
                                )
                                _logger.info(
                                    "Copy from Onnx to PyTorch: torch : %s  matmul bias from onnx param : %s",
                                    name,
                                    found_param_name,
                                )
                            else:
                                if module.weight.shape != onnx_wt_or_bias_shape:
                                    raise RuntimeError(
                                        f"pt weight shape {module.weight.shape} did not match onnx shape {onnx_wt_or_bias_shape}"
                                    )
                                module.weight = torch_param
                                self.mapping_pt_to_onnx_params[name]["weight"] = (
                                    found_param_name
                                )
                                _logger.info(
                                    "Copy from Onnx to PyTorch: torch : %s  layernorm weight from onnx param : %s",
                                    name,
                                    found_param_name,
                                )
                        else:
                            raise ValueError(
                                f"Onnx and pyTorch layer parameter shape is not matching for {name}"
                            )
        return pt_block

    def _climb_parent(self, edge: str):
        return self.name_to_node[self.node_edge_to_parent[edge]]

    def _get_wt_bias_param(self, edge: str):
        """
        Assumption: wt or bias param will be input to QcQuantizeOp optype
        climb up 2 step for a given edge to check for wts or bias
        """
        init_key = None
        parent1_node = self._climb_parent(edge)
        if parent1_node.op_type == "QcQuantizeOp":
            if parent1_node.input[0] in self.initializer_name_to_index_map:
                init_key = parent1_node.input[0]
            if not init_key:
                parent2_node = self._climb_parent(parent1_node.input[0])
                if parent2_node.input[0] in self.initializer_name_to_index_map:
                    init_key = parent2_node.input[0]
        return init_key

    def _copy_weights_encodings_pt_to_onnx(self, pt_block: torch.nn.Module):
        """
        Given a pt_block with adascale params computed, copy the params to onnx model
        """
        for name, module in pt_block.named_modules():
            if isinstance(module, QuantizedLinear):
                pytorch_weight = (
                    module.param_quantizers["weight"]
                    .get_folded_weight(module.weight)
                    .detach()
                    .cpu()
                    .numpy()
                )

                onnx_param_name = self.mapping_pt_to_onnx_params[name]["weight"]
                onnx_initializer_list_idx = self.initializer_name_to_index_map[
                    onnx_param_name
                ]
                onnx_param_tensor = numpy_helper.to_array(
                    self.qsim.model.model.graph.initializer[onnx_initializer_list_idx]
                )

                if name in self.needs_transpose:
                    pytorch_weight = pytorch_weight.T

                if pytorch_weight.shape != onnx_param_tensor.shape:
                    raise ValueError(
                        f"pt param shape {pytorch_weight.shape} did not match onnx shape {onnx_param_tensor.shape}"
                    )

                if not (pytorch_weight == onnx_param_tensor).all():
                    _logger.info(
                        "Copy from PyTorch to ONNX: torch : %s  onnx param : %s",
                        name,
                        onnx_param_name,
                    )

                    if (
                        self.qsim.qc_quantize_op_dict[
                            onnx_param_name
                        ].quant_info.blockSize
                        != 0
                    ):
                        raise RuntimeError("AdaScale with BQ is not supported")

                    self.qsim.model.model.graph.initializer[
                        onnx_initializer_list_idx
                    ].CopyFrom(numpy_helper.from_array(pytorch_weight, onnx_param_name))

                    # copy encodings over to onnx quantizers
                    new_scales = (
                        module.param_quantizers["weight"]
                        .get_scale()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    new_offsets = (
                        module.param_quantizers["weight"]
                        .get_offset()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    new_min = (
                        module.param_quantizers["weight"]
                        .get_min()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    new_max = (
                        module.param_quantizers["weight"]
                        .get_max()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    enc = self.qsim.qc_quantize_op_dict[onnx_param_name].get_encodings()
                    if (
                        len(new_scales) != len(enc)
                        or len(new_offsets) != len(enc)
                        or len(new_min) != len(enc)
                        or len(new_max) != len(enc)
                    ):
                        raise RuntimeError(
                            "Encodings of the onnx quantizer and adascale quantizer have different lengths"
                        )
                    for i, encoding in enumerate(enc):
                        encoding.delta = new_scales[i]
                        encoding.offset = new_offsets[i]
                        encoding.min = new_min[i]
                        encoding.max = new_max[i]

                    self.qsim.qc_quantize_op_dict[onnx_param_name].load_encodings(enc)
