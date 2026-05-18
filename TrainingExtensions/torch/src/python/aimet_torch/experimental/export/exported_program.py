# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access
from contextlib import contextmanager
from typing import Optional, Callable
import re
import torch
from torch._C import _fx_map_arg
from torch.export.graph_signature import TensorArgument
import aimet_torch
from aimet_torch.quantization.affine import QuantizeDequantize
from .export import _post_process
from ._utils import (
    _is_grid_preserving_op,
    _remove_dangling_nodes,
    _eval_node,
    _insert_placeholder,
    _is_multi_output_op,
    _refresh_output_specs,
)


class ExportedProgram(torch.export.ExportedProgram):
    @classmethod
    def from_torch_exported_program(cls, ep: torch.export.ExportedProgram):
        new_ep = cls.__new__(cls)
        new_ep.__dict__ = ep.__dict__
        return new_ep

    def run_decompositions(
        self,
        decomp_table: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
        decompose_custom_triton_ops: bool = False,
    ) -> "ExportedProgram":
        ep = super().run_decompositions(decomp_table, decompose_custom_triton_ops)
        # super().run_decompositions returns the base ExportedProgram classs.
        # Cast it back to subclass to access additional methods and properties
        return self.from_torch_exported_program(ep)

    @contextmanager
    def compute_missing_encodings(self, param_bw: int, activation_bw: int):
        """
        Add missing output quantizers and compute encodings.

        NOTE: This is prototype hardcoded with W8A16

        Example:

            >>> # *Partially* quantized aten graph
            >>> ep = aimet_torch.experimental.export.export(sim.model, (dummy_input,))
            >>> torch.export.save(ep, "partially_quantized_model.pt2")
            >>> # Add missing quantizers and compute encodings
            >>> with ep.compute_missing_encodings(param_bw=8, activation_bw=16):
            ...     for inp in calibration_dataloader:
            ...         ep.module()(inp)
            ...
            >>> # *Fully* quantized aten graph
            >>> torch.export.save(ep, "fully_quantized_model.pt2")
        """
        # Step 1. Lower Aten to core Aten.
        #
        # At this stage, the set of decomposition rules should be selected carefully
        # because some of the default decompositions are too aggressive for QAIRT/HTP.
        #
        # For example, default decomposition table contains the following entry:
        #   ``prelu(x, c) -> where(x >= 0, x, x * c)``
        #
        # Adding intermediate output QDQ to the decomposed prelu subgraph graph will harm
        # both accuracy and performance on QAIRT/HTP.
        #
        # Here, we put an empty decomposition table as a temporary placeholder
        # TODO: Fill out decomposition table
        self.run_decompositions({})

        # Step 2. Add missing quantizers
        #
        # Add QuantizeDequantize module after
        # every floating-point op doesn't have quantized outputs.
        newly_added_qtzrs = self._add_missing_quantizers(param_bw, activation_bw)

        # Step 3. Enter compute_encodings mode and yield control back to user to
        # calibrate encodings of the newly added quantizers. While the control
        # flow is yielded, the user is expected to run forward passes with
        # calibration dataset
        with aimet_torch.nn.compute_encodings(self.module()):
            yield

        # Step 4. Inline QuantizeDequantize modules into torch.ao Q/DQ operators
        # (torch.ops.quantized_decomposed.(de)quantize_per_*)
        self._inline_qdq(newly_added_qtzrs)

        # Step 5. Post-processing
        _post_process(self)
        self._fold_param_qantizers()
        _remove_dangling_nodes(self)

        print(
            "Added and calibrated encodings for the following missing quantizers:"
            f"\n{list(newly_added_qtzrs.keys())}"
        )

    @_refresh_output_specs
    def _add_missing_quantizers(
        self, param_bw: int, activation_bw: int
    ) -> dict[str, QuantizeDequantize]:
        graph_module = self.graph_module
        graph = graph_module.graph

        newly_added_qtzrs: dict[str, QuantizeDequantize] = {}

        def replace(args, old: torch.fx.Node, new: torch.fx.Node):
            return _fx_map_arg(args, lambda arg: new if arg is old else arg)

        for node in graph.nodes:
            # Exclude grid-preserving ops (aka data movement ops) if its input is already quantized.
            # This is to avoid redundant quantize-dequantize pairs around data movement ops.
            if _is_grid_preserving_op(node):
                continue

            if _is_multi_output_op(node):
                continue

            tensor_meta = node.meta.get("tensor_meta", None)
            val = node.meta.get("val", None)

            # Exclude non-tensors from quantization
            if tensor_meta is None or not isinstance(val, torch.Tensor):
                continue

            # Exclude non-floating point tensors from quantization
            if not tensor_meta.dtype.is_floating_point:
                continue

            if node.name in newly_added_qtzrs:
                continue

            if _is_torch_ao_qdq_node(node):
                continue

            if all(_is_torch_ao_qdq_node(user) for user in node.users):
                continue

            if self._is_static_tensor(node):
                # TODO: Skip bias encoding calibration and derive it analytically
                #       from weight and activation encodings
                if _is_bias(node):  # bias
                    bitwidth = 32
                    symmetric = True
                else:  # weight
                    bitwidth = param_bw
                    symmetric = True
            else:  # activation
                bitwidth = activation_bw
                symmetric = False

            output_qdq = QuantizeDequantize((), bitwidth=bitwidth, symmetric=symmetric)
            output_qdq.to(node.meta["val"].device)
            graph_module.add_module(f"{node.name}_qdq", output_qdq)

            with graph.inserting_after(node):
                output_qdq_node = graph.call_module(f"{node.name}_qdq", args=(node,))

            for user in list(node.users):
                if user is output_qdq_node:
                    continue

                if _is_torch_ao_qdq_node(user):
                    continue

                user.args = replace(user.args, old=node, new=output_qdq_node)
                user.kwargs = replace(user.kwargs, old=node, new=output_qdq_node)

            newly_added_qtzrs[output_qdq_node.name] = output_qdq

        # Recompile the graph
        graph_module.recompile()
        return newly_added_qtzrs

    @torch.no_grad()
    @_refresh_output_specs
    def _inline_qdq(self, newly_added_qtzrs: dict[str, QuantizeDequantize]):
        """
        Inline AIMET QuantizeDequantize nodes into torch.ops.quantized_decomposed.(de)quantize_per_*
        """
        uncalibrated_qtzrs = [
            name
            for name, qtzr in newly_added_qtzrs.items()
            if not qtzr.is_initialized()
        ]

        if uncalibrated_qtzrs:
            raise RuntimeError(
                "Following quantizers are not calibrated:\n{uncalibrated_qtzrs}\n"
                "Please make sure to run forward passes with "
                "calibration dataset inside the context manager."
            )

        from torch.fx.passes.shape_prop import _extract_tensor_metadata
        from aimet_torch.quantization.affine.backends.torch_builtins import _get_dtype

        def _inline_qdq(
            qdq_node: torch.fx.Node,
            scale: torch.Tensor,
            zero_point: torch.Tensor,
            qmin: int,
            qmax: int,
        ):
            dtype = _get_dtype(qmin, qmax)

            if scale.dim() == zero_point.dim() == 0:
                scale = scale.item()
                zero_point = zero_point.item()
            else:
                raise NotImplementedError(
                    "Per-channel quantization is not supported yet"
                )

            (input_node,) = qdq_node.all_input_nodes

            with self.graph.inserting_after(qdq_node):
                q_node = self.graph.create_node(
                    op="call_function",
                    target=torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    args=(*qdq_node.args, scale, zero_point, qmin, qmax, dtype),
                    kwargs=qdq_node.kwargs.copy(),
                    name=f"{input_node.name}_q",
                )
                q_node.meta["val"] = input_node.meta["val"].to(dtype, copy=True)
                q_node.meta["tensor_meta"] = _extract_tensor_metadata(
                    q_node.meta["val"]
                )

            with self.graph.inserting_after(q_node):
                dq_node = self.graph.create_node(
                    op="call_function",
                    target=torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    args=(q_node, scale, zero_point, qmin, qmax, dtype),
                    kwargs=qdq_node.kwargs.copy(),
                    name=f"{input_node.name}_dq",
                )
                dq_node.meta["val"] = input_node.meta["val"].clone()
                dq_node.meta["tensor_meta"] = _extract_tensor_metadata(
                    dq_node.meta["val"]
                )

            qdq_node.replace_all_uses_with(dq_node)
            self.graph.erase_node(qdq_node)

        for qdq_node in self.graph_module.graph.nodes:
            if qdq_node.name not in newly_added_qtzrs:
                continue

            qtzr = newly_added_qtzrs[qdq_node.name]
            encoding = qtzr.get_encodings()
            scale = encoding.scale
            zero_point = -encoding.offset.to(torch.int32)
            _inline_qdq(qdq_node, scale, zero_point, encoding.qmin, encoding.qmax)

    def _is_static_tensor(self, node: torch.fx.Node) -> bool:
        if node.op != "placeholder":
            return False

        try:
            param_name = next(
                input_spec.target
                for input_spec in self.graph_signature.input_specs
                if isinstance(input_spec.arg, TensorArgument)
                and input_spec.arg.name == node.name
            )
        except StopIteration:
            return False

        return param_name in self.state_dict

    def _fold_param_qantizers(self):
        for node in self.graph_module.graph.nodes:
            if self._is_static_tensor(node):
                self._do_fold_param_qantizers(node)

    def _do_fold_param_qantizers(self, node: torch.fx.Node):
        if not self._is_static_tensor(node):
            raise RuntimeError

        if len(node.users) != 1:
            return

        (q,) = node.users

        if not _is_torch_ao_qdq_node(q):
            return

        if len(q.users) != 1:
            return

        (dq,) = q.users

        if not _is_torch_ao_qdq_node(dq):
            return

        Wq: torch.Tensor = _eval_node(dq.all_input_nodes[0], self)
        _, dtype_str = str(Wq.dtype).split(".")

        Wq_placeholder: torch.fx.Node = _insert_placeholder(
            self,
            val=Wq,
            node_name=f"{node.name}_{dtype_str}",
            tensor_name=f"{node.name}_{dtype_str}",
            consumers=[dq],
        )
        q.replace_all_uses_with(Wq_placeholder)


def _is_bias(node: torch.fx.Node) -> bool:
    return node.name.endswith("bias")  # TODO


def _is_torch_ao_qdq_node(node: torch.fx.Node) -> bool:
    return bool(
        isinstance(node.target, torch._ops.OpOverload)
        and re.match(
            r"quantized_decomposed::(de)?quantize_per_.+",
            node.target.name(),
        )
    )
