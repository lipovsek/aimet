# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""Sequential MSE implementation"""

# pylint: disable=no-name-in-module, ungrouped-imports, too-many-lines

from typing import List, Dict, Collection, Union, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager
import itertools
import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime
import torch
from onnxruntime.quantization.onnx_quantizer import ONNXModel

from aimet_onnx.common.libpymo import TensorQuantizerOpMode
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.common.utils import AimetLogger, deprecated
from aimet_onnx.qc_quantize_op import GroupedBlockQuantizeDequantize
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.sequential_mse.dependency_graph import (
    DependencyGraph,
    SUPPORTED_MODULES,
)
from aimet_onnx.utils import (
    disable_quantizers,
    OrtInferenceSession,
    get_torch_device,
    map_np_dtype_to_torch,
    LazyExtractor,
)
from aimet_onnx.sequential_mse.dependency_graph import DependencyNode
from aimet_onnx.sequential_mse.transform import (
    modify_graph_with_grouped_conv,
    modify_graph_with_grouped_linear,
    prepare_linear_inputs,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


def apply_seq_mse(
    sim: QuantizationSimModel,
    inputs: Collection[Dict[str, np.ndarray]],
    num_candidates: int = 20,
    nodes_to_exclude: Optional[List[str]] = None,
):
    """
    Sequentially optimizes the QuantizationSimModel's weight encodings to reduce MSE loss at layer outputs.

    Args:
        sim (QuantizationSimModel): QuantizationSimModel instance to optimize
        inputs (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization
        num_candidates (int): Number of encoding candidates to sweep for each weight. Decreasing this can reduce
            runtime but may lead to lower accuracy.
        nodes_to_exclude (Optional[List[str]]): List of supported node name(s) to exclude from sequential MSE optimization
    """
    seq_mse_params = SeqMseParams(num_batches=None, num_candidates=num_candidates)
    seq_mse = SequentialMse(None, sim, seq_mse_params, inputs, nodes_to_exclude)
    seq_mse.apply_seq_mse_algo()


@dataclass
class SeqMseParams:
    """
    Sequential MSE parameters

    :param num_batches: Number of batches.
    :param num_candidates: Number of candidates to perform grid search. Default 20.
    :param inp_symmetry: Input symmetry. Available options are 'asym', 'symfp' and 'symqt'. Default 'symqt'.
    :param loss_fn: Loss function. Available options are 'mse', 'l1' and 'sqnr'. Default 'mse'.
    """

    num_batches: int
    num_candidates: int = 20
    inp_symmetry: str = "symqt"
    loss_fn: str = "mse"


# pylint: disable=too-many-instance-attributes
class SequentialMse:
    """
    Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        sim: QuantizationSimModel,
        params: SeqMseParams,
        data_loader: Collection[Dict[str, np.ndarray]],
        nodes_to_exclude: Optional[List[str]] = None,
    ):
        """
        Initialize the sequential mse object

        :param model: float model
        :param sim: QuantizationSimModel object
        :param params: Sequential MSE parameters
        :param data_loader: The set of input samples to use during optimization
        nodes_to_exclude: List of supported node name(s) to exclude from sequential MSE optimization
        """
        # pylint: disable=protected-access
        assert sim._quant_scheme in (
            QuantScheme.post_training_tf,
            QuantScheme.training_range_learning_with_tf_init,
        ), "Use TF quant-scheme with sequential MSE."

        self.sim = sim
        self.params = params
        self._nodes_to_exclude = nodes_to_exclude or []

        data_loader = itertools.islice(data_loader, params.num_batches)
        # As of onnx 1.18, value info must be populated prior to instantiating Extractor
        with _add_value_info(sim.model.model):
            # For onnx < 1.18, must disable shape inference or it will fail due to custom ops
            with _disable_onnx_shape_inference():
                self._extractor = LazyExtractor(sim.model.model)

        self.dependency_graph = DependencyGraph(
            sim.connected_graph, data_loader, nodes_to_exclude
        )
        self.data_loader = data_loader

    @deprecated("Use aimet_onnx.apply_seq_mse instead")
    @staticmethod
    def apply_seq_mse(
        model: onnx.ModelProto,
        sim: QuantizationSimModel,
        params: SeqMseParams,
        data_loader: Collection[Dict[str, np.ndarray]],
    ):
        """
        It performs following steps:
        1) creates seq_mse object
        2) call apply_seq_algo() member function

        :param model: float model
        :param sim: QuantizationSimModel object
        :param params: Sequential MSE parameters
        :param data_loader: The set of input samples to use during optimization
        """
        seq_mse = SequentialMse(model, sim, params, data_loader)
        seq_mse.apply_seq_mse_algo()

    def apply_seq_mse_algo(self):
        """
        It performs following steps:
        1) disable the quantizer for unsupported modules
        2) create the dependency graph
        3) run the onnx graph and compute encoding using seq mse algorithm
        4) re-enable the quantizer disabled in first step
        """
        with _temporarily_disable_block_grouping(self.sim):
            self.sim._compute_param_encodings(overwrite=False)

            with (
                disable_quantizers(self.sim, self._get_quantizers_to_be_disabled()),
                _remove_session(self.sim),
            ):
                self._topological_traversal()

    def _get_quantizers_to_be_disabled(self) -> List[str]:
        """
        Get list of quantizer names to be disabled in sim model before applying seq mse.

        NOTE: Disable all activation quantizers and param quantizers of non-supported modules

        :return Returns the quantizer names to be disabled in sim model.
        """
        enabled_quantizer_names = []

        # Get list of all the enabled activation + param quantizer names
        for name, quantizer in self.sim.qc_quantize_op_dict.items():
            if quantizer.enabled:
                enabled_quantizer_names.append(name)

        # Get list of all the enabled param quantizers of supported ops
        param_quantizer_names = []
        for cg_op in self.dependency_graph.conn_graph.ordered_ops:
            if cg_op.type not in SUPPORTED_MODULES:
                continue

            if cg_op.name in self._nodes_to_exclude:
                continue

            for param_name in cg_op.parameters:
                quantizer = self.sim.qc_quantize_op_dict.get(param_name)
                if quantizer and quantizer.enabled:
                    param_quantizer_names.append(param_name)

        # Get list of all the quantizers that are not part of param quantizers of supported ops
        quantizers_to_be_disabled = []
        for name in enabled_quantizer_names:
            if name not in param_quantizer_names:
                quantizers_to_be_disabled.append(name)

        return quantizers_to_be_disabled

    @contextmanager
    def _disable_subgraph_quantizers(self, model: onnx.ModelProto):
        quantizer_keys = [
            node.input[0] for node in model.graph.node if node.op_type == "QcQuantizeOp"
        ]
        enabled = {
            name: self.sim.qc_quantize_op_dict[name].enabled for name in quantizer_keys
        }
        try:
            for name in quantizer_keys:
                self.sim.qc_quantize_op_dict[name].enabled = False

            yield

        finally:
            for name in quantizer_keys:
                self.sim.qc_quantize_op_dict[name].enabled = enabled[name]

    def _get_min_and_max_for_candidate_selection(
        self,
        dependency_node: DependencyNode,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recalibrated min/max values for candidate selection.

        :param dependency_node: Dependency node which is to be optimized
        :return: Tuple of min and max values for candidate selection.
        """
        # pylint: disable=protected-access

        weight_name = self.dependency_graph.get_param_name(dependency_node)

        # Retrieve the quantizer and its encodings
        quantizer = self.sim.qc_quantize_op_dict[weight_name]
        encodings = quantizer.get_encodings()
        encoding_shape = quantizer._encoding_shape()

        min_tensor = np.array([enc.min for enc in encodings], dtype=np.float32)
        max_tensor = np.array([enc.max for enc in encodings], dtype=np.float32)

        # Reshape if encoding_shape is not scalar
        if encoding_shape:
            min_tensor = min_tensor.reshape(encoding_shape)
            max_tensor = max_tensor.reshape(encoding_shape)

        return min_tensor, max_tensor

    def _get_candidate(
        self,
        candidate_idx: Union[int, np.ndarray],
        min_tensor: np.ndarray,
        max_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get candidate min and max tensors on the fly.

        NOTE: Divides `min_tensor` and `max_tensor` into `num_candidates` equal parts and
              select the values corresponding to the `candidate_idx + 1`

        :param min_tensor: Min tensor
        :param max_tensor: Max tensor
        :return: candidates
        """
        num_candidates = self.params.num_candidates

        cand_min = min_tensor / num_candidates * (candidate_idx + 1)
        cand_max = max_tensor / num_candidates * (candidate_idx + 1)

        return cand_min, cand_max

    def _compute_encoding_from_candidate(
        self,
        dependency_node: DependencyNode,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ):
        """
        Computes the encoding using candidate min and candidate max values.

        :param dependency_node: Corresponding Dependency node
        :param x_min: min values
        :param x_max: max values
        """
        # Get the parameter name and corresponding quantizer
        weight_name = self.dependency_graph.get_param_name(dependency_node)
        quantizer = self.sim.qc_quantize_op_dict[weight_name]

        # Extract quantization configuration
        channel_axis = quantizer.quant_info.channelAxis
        block_size = quantizer.quant_info.blockSize

        if np.isscalar(x_min) and np.isscalar(x_max):
            x_min = np.asarray([x_min], dtype=np.float32)
            x_max = np.asarray([x_max], dtype=np.float32)

        # Stack x_min and x_max to form candidate tensor
        if block_size > 0:
            # For per-block quantization, always stack along axis=0 -> shape: (2, ...)
            cand = np.stack((x_min, x_max), axis=0)

        # For per-tensor/per-channel, stack based on channel axis
        else:
            if channel_axis == 0:
                # Channels along axis 0 -> shape: (num_channels, 2)
                cand = np.stack((x_min, x_max), axis=1)
            elif channel_axis == 1:
                # Channels along axis 1 -> shape: (2, num_channels)
                cand = np.stack((x_min, x_max), axis=0)
            else:
                raise ValueError(f"Unsupported channel_axis: {channel_axis}")

        # Reset and update encodings statistics with the candidate tensor
        quantizer.reset_encoding_stats()
        quantizer.update_encoding_stats(cand)

        # Compute final encodings and set QDQ mode
        quantizer.compute_encodings()
        quantizer.op_mode = TensorQuantizerOpMode.quantizeDequantize

    def _freeze_encodings(self, dependency_node: DependencyNode):
        """
        Freezes the encoding after the node is optimized
        :param dependency_node: Optimized dependency node
        """
        weight_name = self.dependency_graph.get_param_name(dependency_node)
        quantize_op = self.sim.qc_quantize_op_dict[weight_name]
        quantize_op.freeze_encodings()

    @staticmethod
    def neg_sqnr(pred: torch.Tensor, target: torch.Tensor, eps=1e-10, reduction="none"):
        """
        Loss function to minimize negative SQNR which is equivalent to maximizing SQNR.

        :param pred: X^Q^ quantized-dequantized values
        :param target: XW FP32 values
        :param eps: epsilon
        :param reduction: unused arg added only to have the same signature as that of functional losses of pytorch library
        :return: Negative SQNR
        """
        # pylint: disable=unused-argument
        quant_error = target - pred
        exp_noise = torch.mean(quant_error**2, 0, keepdim=True) + eps
        exp_signal = torch.mean(target**2, 0, keepdim=True)
        sqnr = exp_signal / exp_noise
        sqnr_db = 10 * torch.log10(sqnr)
        return -sqnr_db

    def _compute_recon_loss(
        self,
        fp_outputs: torch.Tensor,
        sim_outputs: torch.Tensor,
        dependency_node: DependencyNode,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss and return the sum by reducing over all the dimensions except last channel dimension.

        :param fp_outputs: fp_outputs (X^W)
        :param sim_outputs: sim_outputs (X^W^)
        :param dependency_node: Dependency node
        :return: loss
        """
        xqwq = sim_outputs
        xqw = fp_outputs

        loss_fn_map = {
            "mse": torch.nn.functional.mse_loss,
            "l1": torch.nn.functional.l1_loss,
            "sqnr": SequentialMse.neg_sqnr,
        }
        if self.params.loss_fn not in loss_fn_map:
            raise ValueError(f"Invalid loss function: {self.params.loss_fn}")

        loss_fn = loss_fn_map[self.params.loss_fn]

        weight_name = self.dependency_graph.get_param_name(dependency_node)
        quantizer = self.sim.qc_quantize_op_dict[weight_name]

        if quantizer is None:
            raise KeyError(f"Quantizer not found for {weight_name}")

        tensor_shape = quantizer.tensor_quantizer_params.tensor_shape
        block_size = quantizer.quant_info.blockSize
        channel_axis = quantizer.quant_info.channelAxis
        block_axis = quantizer.quant_info.blockAxis

        # Handle block-wise quantization
        if block_size > 0:
            if dependency_node.cg_op.type == "Conv":
                loss = loss_fn(xqwq, xqw, reduction="none").sum(dim=(0, 2, 3))
                loss = loss.reshape(tensor_shape[block_axis] // block_size, -1)
                loss = loss.permute(1, 0)
            elif dependency_node.cg_op.type in ["Gemm", "MatMul"]:
                loss = loss_fn(xqwq, xqw, reduction="none").sum(dim=1)
                if block_axis > channel_axis:
                    loss = loss.permute(1, 0)  # For transposed form of Gemm
            else:
                raise NotImplementedError(
                    f"Unsupported op type with block quantization: {dependency_node.cg_op.type}"
                )
            return loss

        # Handle per-tensor and per-channel case
        if dependency_node.cg_op.type == "Conv":
            # Permute to move channel dimension to the end
            xqwq = xqwq.transpose(1, xqwq.dim() - 1)
            xqw = xqw.transpose(1, xqw.dim() - 1)

        # Flatten all dimensions except channel
        channel_dim = xqwq.shape[-1]
        xqwq = xqwq.reshape(-1, channel_dim)
        xqw = xqw.reshape(-1, channel_dim)

        # Compute channel-wise loss
        loss = loss_fn(xqwq, xqw, reduction="none").sum(0)
        return loss

    def _run_seq_mse(self, dep_nodes_to_parallelize: List[DependencyNode]):
        """
        Run Sequential MSE for all the dep_nodes_to_parallelize at same level.

        :param dep_nodes_to_parallelize: Dependency nodes to be parallelized.
        """

        def _set_candidates(candidate_index: int):
            """
            Helper function to set candidate based on index for ops at same level.
            Internally computes the encoding using candidate min and candidate max

            :param candidate_index: Index of candidate
            """
            for dep_node in dep_nodes_to_parallelize:
                init_min_val, init_max_val = initial_min_max[dep_node.cg_op.name]
                cand_min, cand_max = self._get_candidate(
                    candidate_index, init_min_val, init_max_val
                )
                self._compute_encoding_from_candidate(dep_node, cand_min, cand_max)

        def _compute_loss(
            all_fp_outputs: List[torch.Tensor], all_sim_outputs: List[torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """
            Helper function to compute reconstruction loss for ops at same level.

            :param all_fp_outputs: FP Outputs of all ops at same level
            :param all_sim_outputs: Sim Outputs of all ops at same level
            """
            candidate_loss = {}

            for i, dep_node in enumerate(dep_nodes_to_parallelize):
                loss = self._compute_recon_loss(
                    all_fp_outputs[i], all_sim_outputs[i], dep_node
                )
                candidate_loss[dep_node.cg_op.name] = loss

            return candidate_loss

        def _get_dep_node_io_names(dep_nodes: List[DependencyNode]):
            """
            Helper function to get the input and output names of subgraph.

            :param dep_nodes: List of dependency nodes to be parallelized.
            :return: Subgraph input and output names.
            """
            subgraph_inputs = []
            subgraph_outputs = []
            for dep_node in dep_nodes:
                subgraph_inputs.append(dep_node.op_input_names[0])
                subgraph_outputs.append(dep_node.op_output_names[0])

            subgraph_inputs = list(set(subgraph_inputs))
            return subgraph_inputs, subgraph_outputs

        # Cache the initial min and max values
        initial_min_max = {}
        for dep_node in dep_nodes_to_parallelize:
            init_min_val, init_max_val = self._get_min_and_max_for_candidate_selection(
                dep_node
            )
            initial_min_max[dep_node.cg_op.name] = (init_min_val, init_max_val)

        subgraph_inp_names, subgraph_outs_names = _get_dep_node_io_names(
            dep_nodes_to_parallelize
        )

        # For now, we only expose "symqt" input symmetry.
        assert self.params.inp_symmetry == "symqt", (
            "Only symmetric quantsim inputs ('symqt') are supported."
        )
        sim_inputs = self.dependency_graph.get_sim_data(dep_nodes_to_parallelize)

        subgraph_model = self._split_onnx_graph(
            self._extractor, subgraph_inp_names, subgraph_outs_names
        )

        # Transform graph for bq/lpbq quantizers
        subgraph_model, sim_inputs = self._transform_graph_for_block_quantization(
            subgraph_model, sim_inputs
        )

        dataset_len = len(next(iter(sim_inputs.values())))
        with self._create_session(subgraph_model) as session:
            # Pre-compute output shapes per batch
            all_out_shapes, all_out_dtypes = _infer_out_shapes_and_dtypes(
                session, sim_inputs
            )

            torch_device = get_torch_device(session)
            device_type = torch_device.type
            device_id = torch_device.index if torch_device.index is not None else 0

            output_names = [out.name for out in session.get_outputs()]

            # Store accumulated loss per dep_node across candidates
            total_loss = defaultdict(list)

            for candidate_index in tqdm(
                range(self.params.num_candidates),
                desc="Candidates",
                position=0,
            ):
                _set_candidates(candidate_index)

                # Initialize per dep_node loss accumulator
                accumulated_loss = defaultdict(lambda: 0)

                for batch_idx in range(dataset_len):
                    # Create OrtValues per batch and bind them immediately
                    # to avoid memory reuse issues in ORT
                    binding = session.io_binding()

                    # Bind all inputs
                    for name, data in sim_inputs.items():
                        inp_ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(
                            data[batch_idx],
                            device_type=device_type,
                            device_id=device_id,
                        )
                        binding.bind_input(
                            name=name,
                            device_type=device_type,
                            device_id=device_id,
                            element_type=data[batch_idx].dtype,
                            shape=data[batch_idx].shape,
                            buffer_ptr=inp_ort_value.data_ptr(),
                        )

                    # Bind outputs dynamically
                    shared_outputs = []
                    for out_name, out_shape, out_dtype in zip(
                        output_names,
                        all_out_shapes[batch_idx],
                        all_out_dtypes[batch_idx],
                    ):
                        torch_dtype = map_np_dtype_to_torch(out_dtype)
                        shared_tensor = torch.empty(
                            out_shape, dtype=torch_dtype, device=torch_device
                        )
                        shared_outputs.append(shared_tensor)
                        binding.bind_output(
                            name=out_name,
                            device_type=device_type,
                            device_id=device_id,
                            element_type=out_dtype,
                            shape=out_shape,
                            buffer_ptr=shared_tensor.data_ptr(),
                        )
                    # Run fp inference
                    with self._disable_subgraph_quantizers(subgraph_model):
                        session.run_with_iobinding(binding)
                    fp_outputs = [out.clone() for out in shared_outputs]

                    # Run sim inference
                    session.run_with_iobinding(binding)
                    sim_outputs = [out for out in shared_outputs]

                    batched_loss = _compute_loss(fp_outputs, sim_outputs)
                    for node_name, b_loss in batched_loss.items():
                        accumulated_loss[node_name] += b_loss

                # After all batches, append the accumulated loss for this candidate
                for node_name, loss in accumulated_loss.items():
                    total_loss[node_name].append(loss)

                _logger.debug(f"Finished candidate {candidate_index}")

        del sim_inputs, fp_outputs, sim_outputs, shared_outputs

        # Postprocessing (not vectorized)
        for dep_node in dep_nodes_to_parallelize:
            loss = total_loss[dep_node.cg_op.name]
            stacked_loss = torch.stack(
                loss, dim=0
            )  # Resulting shape depends on granularity: per-tensor and per-channel: (num_candidates, num_channels) and per-block: (num_candidates, num_channels, num_blocks)

            # Find the index of the minimum loss along the candidate axis (axis=0)
            best_indices = stacked_loss.min(0)[1]
            init_min, init_max = initial_min_max[dep_node.cg_op.name]

            # Unsqueeze best_indices until it matches dim length of max_val
            while best_indices.ndim < init_max.ndim:
                best_indices = best_indices[..., None]
            best_indices = best_indices.cpu().numpy()

            # Use the best indices to compute corresponding best candidate min and max tensors
            best_min, best_max = self._get_candidate(best_indices, init_min, init_max)

            # Compute and freeze parameter encodings using best candidate
            self._compute_encoding_from_candidate(dep_node, best_min, best_max)
            self._freeze_encodings(dep_node)

        dep_node_names = [dep_node.cg_op.name for dep_node in dep_nodes_to_parallelize]
        _logger.info(
            f"Computed optimal parameter encodings for ops: {', '.join(dep_node_names)}"
        )

    @staticmethod
    def _split_onnx_graph(
        extractor: LazyExtractor, input_names: List[str], output_names: List[str]
    ) -> onnx.ModelProto:
        """
        Splits the onnx graph from input names to output names using extractor

        :param input_names: input names of split graph
        :param output_names: output names of split graph
        :return: float split model and sim split model
        """
        return extractor.extract_model(list(input_names), list(output_names))

    def _run_onnx_graph(
        self, session: onnxruntime.InferenceSession, inputs: Dict
    ) -> List[List[np.ndarray]]:
        """
        Run the onnx graph using onnx runtime

        :param session: Onnxruntime session
        :param inputs: inputs to the model
        :return: outputs
        """
        outputs = []
        dataset_len = len(next(iter(inputs.values())))
        for i in range(dataset_len):
            input_batch = {}
            for name, data in inputs.items():
                input_batch[name] = data[i]
            output = session.run(None, input_batch)
            if len(outputs) == 0:
                outputs = [[] for _ in range(len(output))]
            for idx, out in enumerate(output):
                outputs[idx].append(out)

        return outputs

    def _cache_subgraph_input_data(self, dep_nodes: List[DependencyNode]):
        """
        For given dependency nodes at the same level, cache intermediate activation data

        - Extract a subgraph using the parent nodes,
        - Collect the intermediate activations by executing the subgraph,
        - Cache these data to provide them to the next subgraph.

        :param dep_nodes: List of dependency nodes at same level
        """
        dep_node_names = [dep_node.cg_op.name for dep_node in dep_nodes]
        _logger.debug(
            f"Started caching inputs for dep nodes: {', '.join(dep_node_names)}"
        )

        subgraph_inp_names, subgraph_out_names = (
            self.dependency_graph.get_subgraph_inp_out_names(dep_nodes)
        )
        subgraph_inps = self.dependency_graph.dependency_node_inputs(dep_nodes)
        assert len(subgraph_inp_names) == len(subgraph_inps)

        _logger.debug(
            f"Subgraph input names: {subgraph_inp_names}, Subgraph output names: {subgraph_out_names}"
        )
        sim_split_model = self._split_onnx_graph(
            self._extractor, subgraph_inp_names, subgraph_out_names
        )
        with self._create_session(sim_split_model) as session:
            subgraph_outs = self._run_onnx_graph(session, subgraph_inps)
        self.dependency_graph.update_sim_data(subgraph_out_names, subgraph_outs)
        _logger.debug(
            f"Collected intermediate data for output names: {subgraph_out_names}"
        )
        del subgraph_inps, subgraph_outs

        # Decrease the reference count for the input data.
        for dep_node in dep_nodes:
            for inward_node in dep_node.inward_nodes:
                inward_node.out_degree = inward_node.out_degree - 1
                if inward_node.out_degree == 0:
                    self.dependency_graph.dec_ref_count(inward_node)

    def _topological_traversal(self):
        """
        Start the topo sort from the starting ops i.e. ops having in_degree equal to zero
        Flow:
            - Cache intermediate activations input data before applying Seq MSE at a given level in topological order.
            - Use cached intermediate activations and run Seq MSE in parallel.

        NOTE: For the first iteration, no need to cache subgraph input data since model graph inputs are already saved.
        """
        sorted_order = self.dependency_graph.get_topologically_sorted_nodes()

        for i, sorted_nodes in sorted_order.items():
            if i != 0:
                self._cache_subgraph_input_data([node for node in sorted_nodes])

            dep_nodes_to_parallelize = [
                node for node in sorted_nodes if node.cg_op.type in SUPPORTED_MODULES
            ]
            if dep_nodes_to_parallelize:
                self._run_seq_mse(dep_nodes_to_parallelize)

    @contextmanager
    def _create_session(self, model: onnx.ModelProto):
        """
        Build and return onnxruntime inference session

        :param model: onnx model
        :return: Session
        """
        try:
            session = OrtInferenceSession(
                model,
                self.sim.providers,
                session_options=self.sim._ort_session_options,
                path=self.sim._path,
            )
            yield session
        finally:
            del session

    def _transform_graph_for_block_quantization(
        self, model: onnx.ModelProto, sim_inputs: Dict
    ):
        """
        Identifies BQ/LPBQ quantizers in the given subgraph and modifies the graph
        to compute block-wise reconstruction loss based on the detected quantization configurations.

        NOTE: If no BQ/LPBQ quantizers are found, the model is returned unchanged.

        Assumptions:
         - If block-wise quantizer(s) are found, all of them should be block-wise quantizers.
         - All block-wise quantizers in the subgraph must have the same `block_size` and `block_axis`.
         - The subgraph must either contain Conv ops or Gemm/MatMul ops, but not both.
         - With the above assumptions satisfied, graph transformation can be performed in a single pass
          across all the relevant ops in the subgraph.

        :param model: Model containing subgraph
        :return: Modified model graph if BQ/LPBQ quantizer are found, else return model as-is.
        """
        param_names = set(self.sim.param_names)

        quantizer_keys = [
            node.input[0]
            for node in model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in param_names
        ]

        # Filter only weight quantizers
        quantizers = {
            name: self.sim.qc_quantize_op_dict[name]
            for name in quantizer_keys
            if self.sim.qc_quantize_op_dict[name].enabled
        }

        bq_quantizers = [
            quantizer
            for quantizer in quantizers.values()
            if quantizer.quant_info.blockSize > 0
        ]

        # Early exit if no block quantizers found
        if not bq_quantizers:
            return model, sim_inputs

        # If some quantizers are block-wise and non block-wise raise an error
        # Add support later
        if len(quantizers) != len(bq_quantizers):
            raise NotImplementedError(
                f"Mixed usage of block-wise and non block-wise quantizers is not supported"
            )

        ref_block_size = bq_quantizers[0].quant_info.blockSize
        ref_block_axis = bq_quantizers[0].quant_info.blockAxis

        for quantizer in bq_quantizers[1:]:
            q_info = quantizer.quant_info
            if q_info.blockAxis != ref_block_axis or q_info.blockSize != ref_block_size:
                raise RuntimeError(
                    f"All block quantizers in the subgraph should have the same quantization configuration"
                )

        conv_ops = {"Conv"}
        linear_ops = {"MatMul", "Gemm"}

        op_types = {node.op_type for node in model.graph.node}
        has_conv = bool(op_types.intersection(conv_ops))
        has_linear = bool(op_types.intersection(linear_ops))

        if has_conv and has_linear:
            raise RuntimeError(f"Subgraph contains both Conv and linear ops.")

        # Transform graph only for block quantization
        if has_conv:
            modify_graph_with_grouped_conv(model, ref_block_size, ref_block_axis)
        elif has_linear:
            sim_inputs = prepare_linear_inputs(sim_inputs, block_size=ref_block_size)
            modify_graph_with_grouped_linear(model, ref_block_size, ref_block_axis)
        else:
            raise RuntimeError(f"Subgraph contains no conv or linear ops.")

        return model, sim_inputs


@contextmanager
def _remove_session(sim: QuantizationSimModel):
    """
    Deletes sim.session for the duration of the context to save GPU memory. Rebuilds the session upon exiting.
    """
    try:
        del sim.session
        yield
    finally:
        sim._rebuild_session()  # pylint:disable = protected-access


@contextmanager
def _disable_onnx_shape_inference():
    infer_shapes = onnx.shape_inference.infer_shapes
    try:
        onnx.shape_inference.infer_shapes = lambda model, *args, **kwargs: model
        yield
    finally:
        onnx.shape_inference.infer_shapes = infer_shapes


@contextmanager
def _remove_initializer_data(model: onnx.ModelProto):
    # Hacky way to get around onnx.shape_inference.infer_shapes call as it doesn't work for model >2GB
    raw_data = {}

    try:
        # Store and clear raw_data from initializers
        for initializer in model.graph.initializer:
            if initializer.HasField("raw_data"):
                raw_data[initializer.name] = initializer.raw_data
                initializer.ClearField("raw_data")

        yield

    finally:
        for initializer in model.graph.initializer:
            if initializer.name in raw_data:
                initializer.raw_data = raw_data[initializer.name]


@contextmanager
def _add_value_info(model: onnx.ModelProto):
    initial_value_info = model.graph.value_info

    # Remove weight data to allow shape inference (fails for models > 2GB)
    with _remove_initializer_data(model):
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(model)

    # Replace quantizers with Identity ops to allow shape inference
    for node in model_copy.graph.node:
        if node.op_type != "QcQuantizeOp":
            continue

        node.op_type = "Identity"
        node.ClearField("attribute")
        node.ClearField("domain")

    # Model must be topologically sorted prior to shape inference
    ONNXModel(model_copy).topological_sort()
    inferred_model = onnx.shape_inference.infer_shapes(model_copy)

    value_info = inferred_model.graph.value_info
    del model_copy, inferred_model

    try:
        # Update model's value info
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(value_info)
        yield
    finally:
        # Restore original value info
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(initial_value_info)


@contextmanager
def _temporarily_disable_block_grouping(sim: QuantizationSimModel):
    """
    Set all grouped block quantizers to regular block-wise quantization for the duration of the context manager.

    NOTE: block grouping of 1 is equivalent to standard block-wise quantization, as each block has its own encodings.

    :param sim: QuantizationSimModel object
    """
    quantizers = [
        sim.qc_quantize_op_dict[name]
        for name in sim.param_names
        if sim.qc_quantize_op_dict[name].enabled
        and isinstance(sim.qc_quantize_op_dict[name], GroupedBlockQuantizeDequantize)
    ]

    original_block_groupings = {
        quantizer: quantizer._block_grouping for quantizer in quantizers
    }

    try:
        for quantizer in quantizers:
            quantizer._block_grouping = lambda q=quantizer: [
                1 for _ in range(len(q._encoding_shape()))
            ]
        yield
    finally:
        for quantizer, original_block_grouping in original_block_groupings.items():
            quantizer._block_grouping = original_block_grouping


def _infer_out_shapes_and_dtypes(
    session: OrtInferenceSession, data: Dict[str, List[torch.Tensor]]
) -> Tuple[List, List]:
    """

    Infers output shapes and data types for each batch of inputs using the provided ONNX Runtime session.

    This function runs the model on each batch of input tensors
    and collects the output shapes and dtypes for later use in output buffer allocation.

    TODO: Instead of running inference, just consider inspecting the graph for dynamic shapes.

    :param session: ORT inference session.
    :param data: Dictionary of input tensors.
    :return: List of output shapes and dtypes per batch of inputs.
    """
    dataset_len = len(next(iter(data.values())))
    all_out_shapes = []
    all_out_dtypes = []

    for batch_idx in range(dataset_len):
        input_batch = {name: array_list[batch_idx] for name, array_list in data.items()}
        outputs = session.run(None, input_batch)
        all_out_shapes.append([out.shape for out in outputs])
        all_out_dtypes.append([out.dtype for out in outputs])
        del outputs

    return all_out_shapes, all_out_dtypes
