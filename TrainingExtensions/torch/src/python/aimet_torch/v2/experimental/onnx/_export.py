# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024-2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Utility APIs for onnx export"""

from contextlib import contextmanager, ExitStack
from collections import defaultdict
from dataclasses import dataclass
from packaging import version
import functools
import math
import numpy as np
from typing import Sequence, Iterable, Optional

import onnx
import onnxscript
from onnxscript import opset15, opset16, opset17, opset18, opset19, opset20, opset21
import torch
from torch.onnx import is_in_onnx_export, symbolic_helper

try:
    # torch <2.9
    from torch.onnx._internal.jit_utils import GraphContext
    from torch.onnx._globals import GLOBALS
except ImportError:
    # torch >=2.9
    from torch.onnx._internal.torchscript_exporter.jit_utils import GraphContext
    from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

from aimet_torch.v2.utils import patch_attr


ONNX_QUANTIZER_OP_TYPES = ("quantize", "quantize_dequantize")
aimet_opset = onnxscript.values.Opset(domain="aimet", version=1)
_is_torch_2 = version.parse(torch.__version__) >= version.parse("2.0.0")


def _quantize_template(opset: onnxscript.values.Opset) -> onnxscript.OnnxFunction:
    @onnxscript.script(aimet_opset, default_opset=opset)
    def quantize(
        tensor, scale, offset, qmin: int, qmax: int, block_size: Sequence[int]
    ):
        """Onnxscript implementation of affine quantize"""
        # Upscale scale/offset by the factor of block_size
        upscaled_shape = opset.Shape(scale) * block_size
        scale = opset.Resize(
            scale, roi=None, scales=None, sizes=upscaled_shape, mode="nearest"
        )

        upscaled_shape = opset.Shape(offset) * block_size
        offset = opset.Resize(
            offset, roi=None, scales=None, sizes=upscaled_shape, mode="nearest"
        )

        x_round = opset.Round(tensor / scale) - offset
        x_int = opset.Clip(x_round, qmin, qmax)
        return opset.Reshape(x_int, opset.Shape(tensor))

    return quantize


def _dequantize_template(opset: onnxscript.values.Opset) -> onnxscript.OnnxFunction:
    @onnxscript.script(aimet_opset, default_opset=opset)
    def dequantize(tensor, scale, offset, block_size: Sequence[int]):
        """Onnxscript implementation of affine dequantize"""
        # Upscale scale/offset by the factor of block_size
        upscaled_shape = opset.Shape(scale) * block_size
        scale = opset.Resize(
            scale, roi=None, scales=None, sizes=upscaled_shape, mode="nearest"
        )

        upscaled_shape = opset.Shape(offset) * block_size
        offset = opset.Resize(
            offset, roi=None, scales=None, sizes=upscaled_shape, mode="nearest"
        )

        x_dq = (tensor + offset) * scale
        return opset.Reshape(x_dq, opset.Shape(tensor))

    return dequantize


def _quantize_dequantize_template(
    opset: onnxscript.values.Opset,
) -> onnxscript.OnnxFunction:
    @onnxscript.script(aimet_opset, default_opset=opset)
    def quantize_dequantize(
        tensor,
        scale,
        offset,
        qmin: int,
        qmax: int,
        block_size: Sequence[int],
        zero_point_shift: float,
    ):
        """Onnxscript implementation of affine quantize-dequantize"""
        # Upscale scale/offset by the factor of block_size
        upscaled_shape = opset.Shape(scale) * block_size
        scale = opset.Resize(
            scale, roi=None, scales=None, sizes=upscaled_shape, mode="nearest"
        )

        upscaled_shape = opset.Shape(offset) * block_size
        offset = opset.Resize(
            offset, roi=None, scales=None, sizes=upscaled_shape, mode="nearest"
        )

        x_round = opset.Round(tensor / scale - zero_point_shift) - offset
        x_int = opset.Clip(x_round, qmin, qmax)
        x_dq = (x_int + offset) * scale
        return opset.Reshape(x_dq, opset.Shape(tensor))

    return quantize_dequantize


@dataclass
class _opset:
    quantize: onnxscript.OnnxFunction
    dequantize: onnxscript.OnnxFunction
    quantize_dequantize: onnxscript.OnnxFunction


_opset15 = _opset(
    _quantize_template(opset15),
    _dequantize_template(opset15),
    _quantize_dequantize_template(opset15),
)
_opset16 = _opset(
    _quantize_template(opset16),
    _dequantize_template(opset16),
    _quantize_dequantize_template(opset16),
)
_opset17 = _opset(
    _quantize_template(opset17),
    _dequantize_template(opset17),
    _quantize_dequantize_template(opset17),
)
_opset18 = _opset(
    _quantize_template(opset18),
    _dequantize_template(opset18),
    _quantize_dequantize_template(opset18),
)
_opset19 = _opset(
    _quantize_template(opset19),
    _dequantize_template(opset19),
    _quantize_dequantize_template(opset19),
)
_opset20 = _opset(
    _quantize_template(opset20),
    _dequantize_template(opset20),
    _quantize_dequantize_template(opset20),
)
_opset21 = _opset(
    _quantize_template(opset21),
    _dequantize_template(opset21),
    _quantize_dequantize_template(opset21),
)


def _unsqueeze_scalar(g, tensor):
    # pylint: disable=protected-access
    shape = symbolic_helper._get_tensor_sizes(tensor) or []
    if len(shape) == 0:
        tensor = symbolic_helper._unsqueeze_helper(g, tensor, [0])
    return tensor


def _shape(tensor):
    return symbolic_helper._get_tensor_sizes(tensor)  # pylint: disable=protected-access


def quantize_symbolic(g, tensor, scale, offset, qmin, qmax, block_size=None):
    """Onnx symbolic function definition for affine quantize"""
    if not _is_torch_2:
        # torch <2 passes torch._C.Graph object instead of GraphContext.
        # Temporarily wrap torch._C.Graph with GraphContext
        from torch.onnx.utils import _params_dict

        g = GraphContext(
            graph=g,
            block=g.block(),
            opset=GLOBALS.export_onnx_opset_version,
            original_node=None,
            params_dict=_params_dict,
            env={},
        )

    # Unsqueeze scale, offset if scalars.
    # This is necessary because ONNX Resize operator requires non-scalar input tensors
    scale = _unsqueeze_scalar(g, scale)
    offset = _unsqueeze_scalar(g, offset)

    if block_size is None:
        block_size = (1,)

    if any(b == -1 for b in block_size):
        # Concretize wildcard block sizes
        old_block_size = block_size
        new_block_size = list(
            reversed(
                [
                    input_dim // num_blocks
                    for input_dim, num_blocks in zip(
                        _shape(tensor)[::-1], _shape(scale)[::-1]
                    )
                ]
            )
        )
        assert all(
            old == new for old, new in zip(old_block_size, new_block_size) if old != -1
        )
        block_size = new_block_size

    if not _is_torch_2:
        # For torch <2, insert dummy placeholder node instead of
        # a runnable onnxscript function since
        # torch 1.x doesn't support adding onnxscript function to onnx graph
        return g.op(
            "aimet::quantize",
            tensor,
            scale,
            offset,
            qmin_i=qmin,
            qmax_i=qmax,
            block_size_i=block_size,
        ).setType(tensor.type())

    opset = (
        _opset15
        if g.opset <= 15
        else _opset16
        if g.opset == 16
        else _opset17
        if g.opset == 17
        else _opset18
        if g.opset == 18
        else _opset19
        if g.opset == 19
        else _opset20
        if g.opset == 20
        else _opset21
    )

    return g.onnxscript_op(
        opset.quantize,
        tensor,
        scale,
        offset,
        qmin_i=qmin,
        qmax_i=qmax,
        block_size_i=block_size,
    ).setType(tensor.type())


def dequantize_symbolic(g, tensor, scale, offset, block_size=None):
    """Onnx symbolic function definition for affine dequantize"""
    if not _is_torch_2:
        # torch <2 passes torch._C.Graph object instead of GraphContext.
        # Temporarily wrap torch._C.Graph with GraphContext
        from torch.onnx.utils import _params_dict

        g = GraphContext(
            graph=g,
            block=g.block(),
            opset=GLOBALS.export_onnx_opset_version,
            original_node=None,
            params_dict=_params_dict,
            env={},
        )

    # Unsqueeze scale, offset if scalars.
    # This is necessary because ONNX Resize operator requires non-scalar input tensors
    scale = _unsqueeze_scalar(g, scale)
    offset = _unsqueeze_scalar(g, offset)

    if block_size is None:
        block_size = (1,)

    if any(b == -1 for b in block_size):
        # Concretize wildcard block sizes
        old_block_size = block_size
        new_block_size = list(
            reversed(
                [
                    input_dim // num_blocks
                    for input_dim, num_blocks in zip(
                        _shape(tensor)[::-1], _shape(scale)[::-1]
                    )
                ]
            )
        )
        assert all(
            old == new for old, new in zip(old_block_size, new_block_size) if old != -1
        )
        block_size = new_block_size

    if not _is_torch_2:
        # For torch <2, insert dummy placeholder node instead of
        # a runnable onnxscript function since
        # torch 1.x doesn't support adding onnxscript function to onnx graph
        return g.op(
            "aimet::dequantize",
            tensor,
            scale,
            offset,
            block_size_i=block_size,
        ).setType(tensor.type())

    opset = (
        _opset15
        if g.opset <= 15
        else _opset16
        if g.opset == 16
        else _opset17
        if g.opset == 17
        else _opset18
        if g.opset == 18
        else _opset19
        if g.opset == 19
        else _opset20
        if g.opset == 20
        else _opset21
    )

    return g.onnxscript_op(
        opset.dequantize, tensor, scale, offset, block_size_i=block_size
    ).setType(tensor.type())


def quantize_dequantize_symbolic(
    g, tensor, scale, offset, qmin, qmax, block_size=None, zero_point_shift=0.0
):
    """Onnx symbolic function definition for affine quantize-dequantize"""
    # Unsqueeze scale, offset if scalars.
    # This is necessary because ONNX Resize operator requires non-scalar input tensors

    if not _is_torch_2:
        # torch <2 passes torch._C.Graph object instead of GraphContext.
        # Temporarily wrap torch._C.Graph with GraphContext
        from torch.onnx.utils import _params_dict

        g = GraphContext(
            graph=g,
            block=g.block(),
            opset=GLOBALS.export_onnx_opset_version,
            original_node=None,
            params_dict=_params_dict,
            env={},
        )

    scale = _unsqueeze_scalar(g, scale)
    offset = _unsqueeze_scalar(g, offset)

    if block_size is None:
        block_size = (1,)

    if any(b == -1 for b in block_size):
        # Concretize wildcard block sizes
        old_block_size = block_size
        new_block_size = list(
            reversed(
                [
                    input_dim // num_blocks
                    for input_dim, num_blocks in zip(
                        _shape(tensor)[::-1], _shape(scale)[::-1]
                    )
                ]
            )
        )
        assert all(
            old == new for old, new in zip(old_block_size, new_block_size) if old != -1
        )
        block_size = new_block_size

    if not _is_torch_2:
        # For torch <2, insert dummy placeholder node instead of
        # a runnable onnxscript function since
        # torch 1.x doesn't support adding onnxscript function to onnx graph
        return g.op(
            "aimet::quantize_dequantize",
            tensor,
            scale,
            offset,
            qmin_i=qmin,
            qmax_i=qmax,
            block_size_i=block_size,
            zero_point_shift_f=zero_point_shift,
        ).setType(tensor.type())

    opset = (
        _opset15
        if g.opset <= 15
        else _opset16
        if g.opset == 16
        else _opset17
        if g.opset == 17
        else _opset18
        if g.opset == 18
        else _opset19
        if g.opset == 19
        else _opset20
        if g.opset == 20
        else _opset21
    )

    return g.onnxscript_op(
        opset.quantize_dequantize,
        tensor,
        scale,
        offset,
        qmin_i=qmin,
        qmax_i=qmax,
        block_size_i=block_size,
        zero_point_shift_f=zero_point_shift,
    ).setType(tensor.type())


def register_symbolic(symbolic_fn):
    """
    Register ONNX symbolic function definition for a regular python function.
    """

    def decorator(python_fn):
        class SymbolicHelper(torch.autograd.Function):  # pylint: disable=abstract-method
            """Helper class for coupling an arbitrary python function with a onnx symbolic function"""

            @staticmethod
            def forward(ctx, *args, **kwargs):  # pylint:disable=arguments-differ, unused-argument
                return python_fn(*args, **kwargs)

            symbolic = staticmethod(symbolic_fn)

        @functools.wraps(python_fn)
        def wrapper(*args, **kwargs):
            if is_in_onnx_export():
                return SymbolicHelper.apply(*args, **kwargs)
            return python_fn(*args, **kwargs)

        return wrapper

    return decorator


def export(model: torch.nn.Module, *args, **kwargs):
    """
    Export a torch model to ONNX with precomputed scale and offset.
    """
    if not isinstance(model, torch.nn.Module):
        raise NotImplementedError

    with _precompute_encodings(model):
        # Precompute scale/offset before entering torch.onnx.export so that
        # scale/offset are always represented as a leaf inputs in the onnx graphs
        return torch.onnx.export(model, *args, **kwargs)


def remove_quantization_nodes_from_onnx_graph(
    model: onnx.ModelProto, base_dir: Optional[str] = None
):  # pylint: disable=too-many-locals, too-many-branches
    """
    Remove quantization nodes from ONNX graph with quantization nodes
    :param model: ONNX model with quantization nodes
    """
    tensor_to_encoding_map = {}
    name_to_producer, name_to_consumer = _get_producer_consumer_info_from_onnx_graph(
        model
    )
    qtzr_nodes = list(
        node for node in model.graph.node if node.op_type in ONNX_QUANTIZER_OP_TYPES
    )

    back_to_back_qdq_tensors = set(qdq.input[0] for qdq in qtzr_nodes) & set(
        qdq.output[0] for qdq in qtzr_nodes
    )
    if back_to_back_qdq_tensors:
        msg = []
        for tensor in back_to_back_qdq_tensors:
            producer = name_to_producer[tensor]
            scale = _get_tensor_from_constant_name(
                model, producer.input[1], base_dir=base_dir
            )
            offset = _get_tensor_from_constant_name(
                model, producer.input[2], base_dir=base_dir
            )
            for consumer in name_to_consumer[tensor]:
                if consumer.op_type not in ONNX_QUANTIZER_OP_TYPES:
                    continue

                if consumer.attribute == producer.attribute:
                    scale_ = _get_tensor_from_constant_name(
                        model, consumer.input[1], base_dir=base_dir
                    )
                    offset_ = _get_tensor_from_constant_name(
                        model, consumer.input[2], base_dir=base_dir
                    )
                    if np.allclose(scale, scale_) and np.all(offset == offset_):
                        # Back-to-back QDQ nodes share same quantization config & parameters.
                        # Tolerate.
                        continue

                msg.append(f"  * {producer.name} -> {consumer.name}")

        if msg:
            raise NotImplementedError(
                "Exporting back-to-back QDQ is not supported. "
                "Detected back-to-back QDQ in the following node sequences:\n"
                + "\n".join(msg)
            )

    graph_outputs = set(output.name for output in model.graph.output)
    for node in qtzr_nodes:
        # Get quantizer name in torch model
        encoding = _get_encoding_from_onnx_node(model, node, base_dir)
        producer = name_to_producer.get(node.input[0])
        consumers = name_to_consumer.get(node.output[0], [])

        if node.output[0] in graph_outputs and not producer:
            # Edge case: This means the model was in form of:
            #   (model_input) --> QDQ -> (model_output)
            # or:
            #   (constant) -----> QDQ -> (model_output)
            #
            # We can't preserve the I/O names in this case
            raise RuntimeError(
                f"Node {node.name} (op_type: {node.op_type}) can't be removed because "
                "it's the only connection between the model's input and output."
            )

        if node.output[0] in graph_outputs:
            # DQ output is part of graph outputs.
            # We should preserve DQ's output name to preserve the graph output name
            #
            # Before:
            #                         +--> consumers
            #   producer -----> QDQ --+--> (model_output)
            #                         ↑
            #                     qdq.output[0]
            #                     (=new_name)
            # After:
            #                         +--> consumers
            #   producer -------------+--> (model_output)
            #                         ↑
            #                     qdq.output[0]
            #                     (=new_name)
            new_name = node.output[0]
        else:
            # Before:
            #   producer -----> Q -> DQ -----> consumers
            #              ↑
            #           q.input[0]
            #          (=new_name)
            # After:
            #   producer --------------------> consumers
            #              ↑
            #           q.input[0]
            #          (=new_name)
            new_name = node.input[0]

        tensor_to_encoding_map[new_name] = encoding

        for consumer in consumers:
            for i, inp in enumerate(consumer.input):
                if inp == node.output[0]:
                    consumer.input[i] = new_name
        if producer:
            for i, out in enumerate(producer.output):
                if out == node.input[0]:
                    producer.output[i] = new_name

    for node in qtzr_nodes:
        # Remove qdq node from graph
        model.graph.node.remove(node)

        # Remove scale and offset from onnx graph
        _remove_constants(model, node.input[1:])

    # Remove custom quantize-dequantize functions since it's not needed anymore
    for func in list(model.functions):
        if func.name in ONNX_QUANTIZER_OP_TYPES:
            model.functions.remove(func)

    # Remove aimet opset from imports since it's not needed anymore
    for opset in model.opset_import:
        if opset.domain == "aimet":
            model.opset_import.remove(opset)
            break

    return tensor_to_encoding_map


def _get_tensor_from_constant_name(
    onnx_model: onnx.ModelProto, constant_name: str, base_dir: Optional[str] = None
):
    """
    Returns tensor from the constant name.
    """
    for node in onnx_model.graph.node:
        if constant_name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return onnx.numpy_helper.to_array(attr.t, base_dir=base_dir)
            raise RuntimeError(
                f"Cannot find value attribute inside constant node {constant_name}"
            )
    raise RuntimeError(f"Cannot find constant with name {constant_name} in onnx model")


def _get_encoding_from_onnx_node(
    onnx_model: onnx.ModelProto,
    quant_node: onnx.NodeProto,
    base_dir: Optional[str] = None,
):
    """
    Get encoding from quantization node.
    """
    # pylint: disable=import-outside-toplevel, protected-access
    from aimet_torch.v2.quantization.affine.encoding import (
        AffineEncoding,
        GroupedBlockEncoding,
    )

    assert quant_node.op_type in ONNX_QUANTIZER_OP_TYPES

    scale, offset = None, None
    qmin, qmax, block_size, zero_point_shift = None, None, None, None
    scale_name, offset_name = quant_node.input[1], quant_node.input[2]

    for attr in quant_node.attribute:
        if attr.name == "qmin":
            qmin = attr.i
        if attr.name == "qmax":
            qmax = attr.i
        if attr.name == "block_size":
            block_size = attr.ints
            if block_size == [1]:
                block_size = None
        if attr.name == "zero_point_shift":
            zero_point_shift = attr.f

        scale = torch.tensor(
            _get_tensor_from_constant_name(onnx_model, scale_name, base_dir)
        )
        offset = torch.tensor(
            _get_tensor_from_constant_name(onnx_model, offset_name, base_dir)
        )

    assert scale is not None
    assert offset is not None

    if scale.numel() == 1 and offset.numel() == 1:
        scale = scale.squeeze()
        offset = offset.squeeze()

    centroid = math.ceil((qmin + qmax) / 2)
    symmetry = bool(torch.all(offset == -centroid))

    encoding = AffineEncoding(
        scale,
        offset,
        qmin,
        qmax,
        symmetry=symmetry,
        block_size=block_size,
        zero_point_shift=zero_point_shift,
    )

    try:
        # Try converting affine encoding to LPBQ encoding if possible
        encoding = GroupedBlockEncoding._from_affine_encoding(encoding)
    except ValueError:
        pass

    return encoding


def _remove_constants(onnx_model: onnx.ModelProto, constant_names: Iterable[str]):
    """
    Remove constants from onnx model.
    """
    constant_names = set(constant_names)
    for node in onnx_model.graph.node[::-1]:
        if node.op_type == "Constant" and node.output[0] in constant_names:
            onnx_model.graph.node.remove(node)


def _get_producer_consumer_info_from_onnx_graph(onnx_model: onnx.ModelProto):
    """
    Get producer and consumer information from ONNX graph for graph traversal.
    :param onnx_model: ONNX model
    :return: Tuple of name to producer mappings and name to consumer mappings
    """
    name_to_producer = {}
    name_to_consumer = defaultdict(list)

    for node in onnx_model.graph.node:
        for output_name in node.output:
            name_to_producer[output_name] = node

        for input_name in node.input:
            name_to_consumer[input_name].append(node)

    return name_to_producer, name_to_consumer


@contextmanager
def _precompute_encodings(model: torch.nn.Module):
    # pylint: disable=import-outside-toplevel
    from aimet_torch.quantization.base import QuantizerBase

    with torch.no_grad():
        encodings = {
            q: q.get_encodings()
            for q in model.modules()
            if isinstance(q, QuantizerBase)
        }

    def is_initialized(q: QuantizerBase):
        return encodings[q] is not None

    def get_cached_encodings(q: QuantizerBase):
        return encodings[q]

    with ExitStack() as stack:
        for q in model.modules():
            if isinstance(q, QuantizerBase):
                ctx = patch_attr(
                    q, "get_encodings", functools.partial(get_cached_encodings, q)
                )
                stack.enter_context(ctx)

                ctx = patch_attr(
                    q, "is_initialized", functools.partial(is_initialized, q)
                )
                stack.enter_context(ctx)
        yield
