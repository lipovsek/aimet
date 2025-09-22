# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from onnx import numpy_helper
import torch
from torch.nn import Parameter
import copy

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

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

    def __init__(self, quantsim, model_config):
        self.qsim = quantsim
        self.model_config = model_config
        self._get_onnx_blocks()
        self._get_pt_model()

    def get_pt_decoder_block(self, idx):
        return self.pt_decoder_blocks[idx]

    def _get_onnx_blocks(self):
        self.end_points = get_decoder_blocks_end_points(self.qsim)
        self.num_layers = len(self.end_points)
        graph = self.qsim.model.model.graph
        self.name_to_node = {n.name: n for n in graph.node}
        self.node_edge_to_parent = {n.output[0]: n.name for n in graph.node}
        self.initializer_map = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }
        self.all_ops = self.qsim.connected_graph.ordered_ops
        self.op_name_to_index = {
            op.name: index for index, op in enumerate(self.all_ops)
        }

    def _copy_weights_onnx_to_pt(self, blk_idx):
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
        pt_block = self.get_pt_decoder_block(blk_idx)

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
            assert len(layer_type) <= 1
            if len(layer_type) == 0:
                continue
            _, elem = layer_type[0]

            target_ops = [
                target_op for target_op in all_ops_filtered if elem in target_op.name
            ]
            assert 1 <= len(target_ops) <= 4, (
                f"We expect between 1 to 4 onnx nodes whose name contains {elem} but got {len(target_ops)}"
            )
            for target_op in target_ops:
                for edge in self.name_to_node[target_op.name].input:
                    found_param_name = self._get_wt_bias_param(edge)
                    if found_param_name:
                        onnx_wt_or_bias = self.initializer_map[found_param_name]
                        onnx_wt_or_bias_shape = onnx_wt_or_bias.shape
                        rank = len(onnx_wt_or_bias_shape)
                        torch_param = copy.deepcopy(
                            Parameter(
                                torch.from_numpy(onnx_wt_or_bias).to(torch.float32)
                            )
                        )
                        if rank == 2:  # its matmul weight
                            pt_wt_shp = module.weight.shape
                            if (
                                onnx_wt_or_bias.shape != pt_wt_shp
                            ):  # try transposing if shapes don't match
                                onnx_wt_or_bias = onnx_wt_or_bias.T
                                onnx_wt_or_bias_shape = onnx_wt_or_bias.shape
                                torch_param = copy.deepcopy(
                                    Parameter(
                                        torch.from_numpy(onnx_wt_or_bias).to(
                                            torch.float32
                                        )
                                    )
                                )
                            assert pt_wt_shp == onnx_wt_or_bias_shape, (
                                f"pt wt shape {pt_wt_shp} did not match onnx shape {onnx_wt_or_bias_shape} (with ot without transpose)"
                            )
                            module.weight = torch_param
                            print(f"Copying wts for {name} from {found_param_name}")
                        elif rank == 1:  # its matmul bias or layernorm wt
                            is_matmul_bias = hasattr(
                                module, "bias"
                            )  # layernorm doesnt have "bias"

                            if is_matmul_bias:
                                assert module.bias.shape == onnx_wt_or_bias_shape
                                module.bias = torch_param
                            else:
                                assert module.weight.shape == onnx_wt_or_bias_shape
                                module.weight = torch_param
                            print(
                                f"Copying {'matmul bias' if is_matmul_bias else 'layernorm wt'} for {name} from {found_param_name}"
                            )

                        else:
                            raise ValueError(
                                f"Onnx and pyTorch layer parameter shape is not matching for {name}"
                            )

    def _climb_parent(self, edge):
        return self.name_to_node[self.node_edge_to_parent[edge]]

    def _get_wt_bias_param(self, edge):
        """
        Assumption: wt or bias param will be input to QcQuantizeOp optype
        climb up 2 step for a given edge to check for wts or bias
        """
        init_key = None
        parent1_node = self._climb_parent(edge)
        if parent1_node.op_type == "QcQuantizeOp":
            if parent1_node.input[0] in self.initializer_map:
                init_key = parent1_node.input[0]
            if not init_key:
                parent2_node = self._climb_parent(parent1_node.input[0])
                if parent2_node.input[0] in self.initializer_map:
                    init_key = parent2_node.input[0]
        return init_key

    def _get_pt_model(self):
        # TODO [ananmukh] - use onnx node info to get the config and remove config hardcoding
        """
        create self.pt_model
        """
        decoder_block_cls = Qwen2DecoderLayer
        config = self.model_config.model_config
        self.pt_decoder_blocks = [
            decoder_block_cls(config, i) for i in range(self.num_layers)
        ]
