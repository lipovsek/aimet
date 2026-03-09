# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utility APIs for onnx export"""

from __future__ import annotations
from contextlib import contextmanager, ExitStack, nullcontext
from collections import defaultdict
from dataclasses import dataclass
from packaging import version
import functools
import math
import numpy as np
import os
from typing import Any, Sequence, Optional, Mapping, TYPE_CHECKING

import onnx
import onnxscript
from onnxscript import opset15, opset16, opset17, opset18, opset19, opset20, opset21
import torch
from torch.onnx import is_in_onnx_export, symbolic_helper
from torch import _C

try:
    # torch <2.9
    from torch.onnx._internal.jit_utils import GraphContext
    from torch.onnx._globals import GLOBALS
except ImportError:
    # torch >=2.9
    from torch.onnx._internal.torchscript_exporter.jit_utils import GraphContext
    from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

from aimet_torch.v2.utils import patch_attr
from aimet_torch.common.onnx._utils import (
    _iterate_graph_nodes_recursive,
    _get_all_constants,
)

if TYPE_CHECKING:
    from aimet_torch.v2.quantization.base.encoding import (
        EncodingBase,
        FloatEncoding,
        AffineEncoding,
    )
    from aimet_torch.v2.quantization.affine import AffineEncoding
    from aimet_torch.v2.quantization.float import FloatEncoding
    from aimet_torch.v2.quantization.float._finfo import _finfo


ONNX_QUANTIZER_OP_TYPES = (
    "quantize",
    "quantize_dequantize",
    "QuantizeDequantize",
    "FloatQuantizeDequantize",
)
aimet_opset = onnxscript.values.Opset(domain="aimet", version=1)
_is_torch_2 = version.parse(torch.__version__) >= version.parse("2.0.0")

# QuantizeDequantize placeholder definition.
# This node is a dummy placeholder for torch.ops.aimet.quantize_dequantize,
# which can't be directly executed by ONNXRuntime.
# These nodes will be substituted with ONNX QuantizeLinear/DequantizeLinear or
# be detached into a separate encoding JSON file
onnx.defs.register_schema(
    onnx.defs.OpSchema(
        name="QuantizeDequantize",
        domain="aimet",
        since_version=1,
        doc="Quantize and Dequantize operation as defined in AIMET.",
        inputs=[
            onnx.defs.OpSchema.FormalParameter(
                name="tensor",
                type_str="T",
                description="Input tensor to be quantized and dequantized",
            ),
            onnx.defs.OpSchema.FormalParameter(
                name="scale",
                type_str="T",
                description="Scale tensor for quantization",
            ),
            onnx.defs.OpSchema.FormalParameter(
                name="offset",
                type_str="T",
                description="Offset tensor for quantization",
            ),
        ],
        outputs=[
            onnx.defs.OpSchema.FormalParameter(
                name="output",
                type_str="T",
                description="Quantized and dequantized output tensor",
            )
        ],
        type_constraints=[
            (
                "T",
                [
                    "tensor(float)",
                    "tensor(double)",
                    "tensor(float16)",
                    "tensor(bfloat16)",
                ],
                "Constrain input and output types to numeric tensors.",
            )
        ],
        attributes=[
            onnx.defs.OpSchema.Attribute(
                name="qmin",
                type=onnx.defs.OpSchema.AttrType.INT,
                description="Minimum quantization value",
                required=True,
            ),
            onnx.defs.OpSchema.Attribute(
                name="qmax",
                type=onnx.defs.OpSchema.AttrType.INT,
                description="Maximum quantization value",
                required=True,
            ),
            onnx.defs.OpSchema.Attribute(
                name="block_size",
                type=onnx.defs.OpSchema.AttrType.INTS,
                description="Block size",
                required=False,
            ),
            onnx.defs.OpSchema.Attribute(
                name="zero_point_shift",
                type=onnx.defs.OpSchema.AttrType.FLOAT,
                description="Zero point shift for quantization",
                required=False,
            ),
        ],
    )
)


onnx.defs.register_schema(
    onnx.defs.OpSchema(
        name="FloatQuantizeDequantize",
        domain="aimet",
        since_version=1,
        doc="Floating-point Quantize and Dequantize operation as defined in AIMET.",
        inputs=[
            onnx.defs.OpSchema.FormalParameter(
                name="tensor",
                type_str="T",
                description="Input tensor to be quantized and dequantized",
            ),
            onnx.defs.OpSchema.FormalParameter(
                name="scale",
                type_str="T",
                description="Scale tensor for quantization",
            ),
        ],
        outputs=[
            onnx.defs.OpSchema.FormalParameter(
                name="output",
                type_str="T",
                description="Quantized and dequantized output tensor",
            )
        ],
        type_constraints=[
            (
                "T",
                [
                    "tensor(float)",
                    "tensor(double)",
                    "tensor(float16)",
                    "tensor(bfloat16)",
                ],
                "Constrain input and output types to numeric tensors.",
            )
        ],
        attributes=[
            onnx.defs.OpSchema.Attribute(
                name="dtype",
                type=onnx.defs.OpSchema.AttrType.INT,
                description="ONNX float4/8 data type. Choices: FLOAT4E2M1, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ",
                required=True,
            ),
            onnx.defs.OpSchema.Attribute(
                name="block_size",
                type=onnx.defs.OpSchema.AttrType.INTS,
                description="Block size",
                required=False,
            ),
        ],
    )
)


@onnxscript.script(aimet_opset, default_opset=onnxscript.opset1)
def _quantize_dequantize_placeholder(
    tensor: onnxscript.FLOAT,
    scale: onnxscript.FLOAT,
    offset: onnxscript.FLOAT,
    qmin: int,
    qmax: int,
    block_size: Sequence[int] = (),
    zero_point_shift: float = 0.0,
) -> onnxscript.FLOAT:
    return aimet_opset.QuantizeDequantize(
        tensor,
        scale,
        offset,
        qmin=qmin,
        qmax=qmax,
        block_size=block_size,
        zero_point_shift=zero_point_shift,
    )


def _quantize_template(opset: onnxscript.values.Opset) -> onnxscript.OnnxFunction:
    @onnxscript.script(aimet_opset, default_opset=opset)
    def quantize(
        tensor, scale, offset, qmin: int, qmax: int, block_size: Sequence[int] = (1,)
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
    def dequantize(tensor, scale, offset, block_size: Sequence[int] = (1,)):
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
        block_size: Sequence[int] = (1,),
        zero_point_shift: float = 0.0,
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

    if block_size is not None:
        from aimet_torch.v2.quantization._utils import concretize_block_size

        block_size = concretize_block_size(_shape(tensor), _shape(scale), block_size)

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

    if block_size is not None:
        from aimet_torch.v2.quantization._utils import concretize_block_size

        block_size = concretize_block_size(_shape(tensor), _shape(scale), block_size)

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

    if block_size is not None:
        from aimet_torch.v2.quantization._utils import concretize_block_size

        block_size = concretize_block_size(_shape(tensor), _shape(scale), block_size)

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


def float_quantize_dequantize_symbolic(
    g, tensor, finfo: _finfo, scale, block_size=None
):
    """Onnx symbolic function definition for affine quantize-dequantize"""
    onnx_float_dtype = finfo.to_onnx_dtype()

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

    if block_size is not None:
        from aimet_torch.v2.quantization._utils import concretize_block_size

        block_size = concretize_block_size(_shape(tensor), _shape(scale), block_size)

    if not _is_torch_2:
        # For torch <2, insert dummy placeholder node instead of
        # a runnable onnxscript function since
        # torch 1.x doesn't support adding onnxscript function to onnx graph
        return g.op(
            "aimet::FloatQuantizeDequantize",
            tensor,
            scale,
            block_size_i=block_size,
            dtype_i=onnx_float_dtype,
        ).setType(tensor.type())

    return g.onnxscript_op(
        aimet_opset.FloatQuantizeDequantize,
        tensor,
        scale,
        dtype_i=onnx_float_dtype,
        block_size_i=block_size,
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


def export(
    model: torch.nn.Module,
    args: tuple[Any, ...] = (),
    f: str | os.PathLike | None = None,
    **kwargs,
):
    """
    Export a torch model to ONNX with precomputed scale and offset.
    """
    if not isinstance(model, torch.nn.Module):
        raise NotImplementedError

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        custom_translation_table = kwargs.get("custom_translation_table", {})
        kwargs["custom_translation_table"] = {
            **custom_translation_table,
            torch.ops.aimet.quantize_dequantize.default: _quantize_dequantize_placeholder,
        }

    with _precompute_encodings(model):
        # Precompute scale/offset before entering torch.onnx.export so that
        # scale/offset are always represented as a leaf inputs in the onnx graphs
        with (
            _disable_torch_C_jit_pass_lint(),
            _maybe_disable_C_jit_pass_onnx_deduplicate_initializers(model),
        ):
            return torch.onnx.export(model, args, f, **kwargs)


@contextmanager
def _disable_torch_C_jit_pass_lint():
    """
    Temporarily disable torch._C._jit_pass_lint.

    This is to work around O(N*B) loop in torch._C._jit_pass_lint,
    where N = # nodes, B = # blocks. While B = 1 for most fp models,
    in most quantized models B ∝ N, leading to O(N^2) time complexity.
    This is because qdq export will contain one autograd.Function ("SymbolicHelper"),
    and torchscript onnx exporter represents each autograd.Function as a single block.

    To avoid this quadratic loop, we disable _C._jit_pass_lint with the following reasoning:

      - _C._jit_pass_lint is just a graph sanity check and does not affect the exported onnx graph
      - There are other types of lint passes for graph sanity check in torch.onnx.export,
        such as _C._jit_pass_onnx_lint
    """
    # TODO: Remove this workaround after torch._C._jit_pass_lint is optimized.
    with patch_attr(_C, "_jit_pass_lint", lambda _: None):
        yield


# Heuristic: Include all LLMs starting from 0.5B models
_LARGE_MODEL_THRESHOLD_NUM_PARAMS = 0.5 * (1024**3)

# Heuristic: Even if it's not an LLM, _C._jit_pass_onnx_deduplicate_initializers
# starts to dominate the export time when the number of nn.Parameter objects
# is larger than certain threshold due to quadratic time complexity.
_LARGE_MODEL_THRESHOLD_NUM_NN_PARAMETER_OBJECTS = 2048


@contextmanager
def _maybe_disable_C_jit_pass_onnx_deduplicate_initializers(model: torch.nn.Module):
    """
    Temporarily disable torch._C._jit_pass_onnx_deduplicate_initializers.

    This is to work around O(N^2) loop in torch._C._jit_pass_onnx_deduplicate_initializers,
    where N = # nn.Parameter objects in the model.

    To avoid this quadratic loop, we disable _C._jit_pass_onnx_deduplicate_initializers
    for large models with the following reasoning:

      - Keeping duplicate initializers doesn't hurt onnx export correctness
      - Duplicate initializers are rare
    """

    # pylint:disable=unused-argument
    def no_op(graph, params_dict, *args, **kwargs):
        return params_dict

    model_size = sum(param.numel() for param in model.parameters())
    num_nn_param_objects = len(list(model.named_parameters(remove_duplicate=False)))

    if (
        model_size >= _LARGE_MODEL_THRESHOLD_NUM_PARAMS
        or num_nn_param_objects >= _LARGE_MODEL_THRESHOLD_NUM_NN_PARAMETER_OBJECTS
    ):
        ctx = patch_attr(_C, "_jit_pass_onnx_deduplicate_initializers", no_op)
    else:
        ctx = nullcontext()

    with ctx:
        yield


def remove_quantization_nodes_from_onnx_graph(
    model: onnx.ModelProto, base_dir: Optional[str] = None
) -> dict[str, EncodingBase]:  # pylint: disable=too-many-locals, too-many-branches
    """
    Remove quantization nodes from ONNX graph with quantization nodes
    :param model: ONNX model with quantization nodes
    """
    tensor_to_encoding_map: dict[str, EncodingBase] = {}
    constants: dict[str, onnx.TensorProto]
    producers: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    qtzr_nodes: dict[str, onnx.NodeProto] = {}

    for node in _iterate_graph_nodes_recursive(model.graph):
        for output_name in node.output:
            producers[output_name] = node

        for input_name in node.input:
            consumers[input_name].append(node)

        if node.domain == "aimet" and node.op_type in ONNX_QUANTIZER_OP_TYPES:
            qtzr_nodes[node.name] = node

    constants = _get_all_constants(model, consumers)

    back_to_back_qdq_tensors = set(qdq.input[0] for qdq in qtzr_nodes.values()) & set(
        qdq.output[0] for qdq in qtzr_nodes.values()
    )
    if back_to_back_qdq_tensors:
        msg = []
        for tensor in back_to_back_qdq_tensors:
            producer = producers[tensor]

            if producer.input[1] in constants:
                scale = onnx.numpy_helper.to_array(
                    constants[producer.input[1]], base_dir=base_dir
                )
            else:
                raise RuntimeError(
                    f"Cannot find constant with name {producer.input[1]} in onnx model"
                )
            if producer.input[2] in constants:
                offset = onnx.numpy_helper.to_array(
                    constants[producer.input[2]], base_dir=base_dir
                )
            else:
                raise RuntimeError(
                    f"Cannot find constant with name {producer.input[2]} in onnx model"
                )

            for consumer in consumers[tensor]:
                if consumer.op_type not in ONNX_QUANTIZER_OP_TYPES:
                    continue

                if consumer.attribute == producer.attribute:
                    if consumer.input[1] in constants:
                        scale_ = onnx.numpy_helper.to_array(
                            constants[consumer.input[1]], base_dir=base_dir
                        )
                    else:
                        raise RuntimeError(
                            f"Cannot find constant with name {consumer.input[1]} in onnx model"
                        )

                    if consumer.input[2] in constants:
                        offset_ = onnx.numpy_helper.to_array(
                            constants[consumer.input[2]], base_dir=base_dir
                        )
                    else:
                        raise RuntimeError(
                            f"Cannot find constant with name {consumer.input[2]} in onnx model"
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

    for node in qtzr_nodes.values():
        # Get quantizer name in torch model
        encoding = _get_encoding_from_onnx_node(constants, node, base_dir)
        producer = producers.get(node.input[0])

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

        for consumer in consumers[node.output[0]]:
            for i, inp in enumerate(consumer.input):
                if inp == node.output[0]:
                    consumer.input[i] = new_name
        if producer:
            for i, out in enumerate(producer.output):
                if out == node.input[0]:
                    producer.output[i] = new_name

    all_nodes = [node for node in model.graph.node if node.name not in qtzr_nodes]
    model.graph.ClearField("node")
    model.graph.node.extend(all_nodes)

    all_functions = [
        func for func in model.functions if func.name not in ONNX_QUANTIZER_OP_TYPES
    ]
    model.ClearField("functions")
    model.functions.extend(all_functions)

    # Remove dangling initializers
    all_input_names = set(inp.name for inp in model.graph.input) | set(
        inp for node in model.graph.node for inp in node.input
    )
    all_initializers = [
        init for init in model.graph.initializer if init.name in all_input_names
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(all_initializers)

    # Remove aimet opset from imports since it's not needed anymore
    all_opset_imports = [
        opset for opset in model.opset_import if opset.domain != "aimet"
    ]
    model.ClearField("opset_import")
    model.opset_import.extend(all_opset_imports)

    return tensor_to_encoding_map


def _get_encoding_from_onnx_node(
    constants: Mapping[str, onnx.TensorProto],
    quant_node: onnx.NodeProto,
    base_dir: Optional[str] = None,
) -> EncodingBase:
    """
    Get encoding from quantization node.
    """
    assert (
        quant_node.domain == "aimet" and quant_node.op_type in ONNX_QUANTIZER_OP_TYPES
    )

    if quant_node.op_type == "FloatQuantizeDequantize":
        return _get_float_encoding_from_onnx_node(
            constants, quant_node, base_dir=base_dir
        )
    else:
        return _get_affine_encoding_from_onnx_node(
            constants, quant_node, base_dir=base_dir
        )


def _get_float_encoding_from_onnx_node(
    constants: Mapping[str, onnx.TensorProto],
    quant_node: onnx.NodeProto,
    base_dir: Optional[str] = None,
) -> FloatEncoding:
    from aimet_torch.v2.quantization.float import FloatEncoding
    from aimet_torch.v2.quantization.float._finfo import _finfo

    finfo = None
    block_size = None

    for attr in quant_node.attribute:
        if attr.name == "dtype":
            finfo = _finfo.from_onnx_dtype(attr.i)
        if attr.name == "block_size":
            block_size = tuple(attr.ints) or None

    if finfo is None:
        raise RuntimeError(
            f"Cannot find 'dtype' attribute in FloatQuantizeDequantize node {quant_node.name}"
        )

    scale_name = quant_node.input[1]

    if scale_name in constants:
        scale = torch.tensor(
            onnx.numpy_helper.to_array(constants[scale_name], base_dir=base_dir)
        )
    else:
        raise RuntimeError(f"Cannot find constant with name {scale_name} in onnx model")

    return FloatEncoding(
        finfo.mantissa_bits,
        finfo.exponent_bits,
        finfo.finite,
        finfo.unsigned_zero,
        scale=scale,
        block_size=block_size,
    )


def _get_affine_encoding_from_onnx_node(
    constants: Mapping[str, onnx.TensorProto],
    quant_node: onnx.NodeProto,
    base_dir: Optional[str] = None,
) -> AffineEncoding:
    # pylint: disable=protected-access
    from aimet_torch.v2.quantization.affine.encoding import (
        AffineEncoding,
        GroupedBlockEncoding,
    )

    qmin, qmax, block_size, zero_point_shift = None, None, None, None
    scale_name, offset_name = quant_node.input[1], quant_node.input[2]
    block_size = None

    for attr in quant_node.attribute:
        if attr.name == "qmin":
            qmin = attr.i
        if attr.name == "qmax":
            qmax = attr.i
        if attr.name == "block_size":
            block_size = tuple(attr.ints) or None
        if attr.name == "zero_point_shift":
            zero_point_shift = attr.f

    if scale_name in constants:
        scale = torch.tensor(
            onnx.numpy_helper.to_array(constants[scale_name], base_dir=base_dir)
        )
    else:
        raise RuntimeError(f"Cannot find constant with name {scale_name} in onnx model")

    if offset_name in constants:
        offset = torch.tensor(
            onnx.numpy_helper.to_array(constants[offset_name], base_dir=base_dir)
        )
    else:
        raise RuntimeError(
            f"Cannot find constant with name {offset_name} in onnx model"
        )

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

    if block_size and symmetry:
        try:
            # Try converting affine encoding to LPBQ encoding if possible
            encoding = GroupedBlockEncoding._from_affine_encoding(encoding)
        except ValueError:
            pass

    return encoding


@torch.no_grad()
@contextmanager
def _precompute_encodings(model: torch.nn.Module):
    # pylint: disable=import-outside-toplevel, protected-access
    from aimet_torch.quantization.base import QuantizerBase

    with ExitStack() as stack:
        for q in model.modules():
            if isinstance(q, QuantizerBase):
                ctx = q._precompute_encodings()
                stack.enter_context(ctx)

        yield
