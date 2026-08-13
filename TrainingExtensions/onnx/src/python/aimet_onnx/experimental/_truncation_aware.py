# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import onnx_ir
from onnx_ir.passes.common import ShapeInferencePass, TopologicalSortPass
import onnxscript
from onnxscript.values import Opset
import numpy as np

from aimet_onnx import QuantizationSimModel
from aimet_onnx.quantsim import QuantizationDataType
from aimet_onnx.ir_utils import remove_aimet_quantizers
from aimet_onnx.qc_quantize_op import EncodingType
import aimet_onnx.utils

_LINEAR_OPS = (
    "Conv",
    "Gemm",
    "MatMul",
    "ConvTranspose",
)


def _is_int_quantizer(quantizer):
    if quantizer is None:
        return False
    if not quantizer.enabled:
        return False
    if not quantizer.is_initialized():
        return False
    return quantizer.data_type == QuantizationDataType.int


_CUSTOM_OP_DOMAIN = "aimet.custom_ops"


def _infer_output_ranks(model: onnx_ir.Model) -> dict:
    """
    Return a ``{tensor_name: rank}`` map for every tensor with an inferred shape.
    """
    model_copy = model.clone()
    remove_aimet_quantizers(model_copy)
    model_copy = ShapeInferencePass()(model_copy).model

    ranks = {}
    for node in model_copy.graph:
        for output in node.outputs:
            if output.shape is not None:
                ranks[output.name] = len(output.shape)
    return ranks


def _insert_output_truncation(
    model: onnx_ir.Model,
    tensor: onnx_ir.Value,
    acc_scale: np.ndarray,
    truncation_bits: int,
):
    """
    Inserts custom Truncate(input, scale) function at tensor_name to simulate int32 accumulator truncation.

    ``acc_scale`` is the int32 accumulator scale shaped to broadcast against ``tensor_name``
    """
    original_name = tensor.name
    scale_name = original_name + "_trunc_scale"
    trunc_scale = (acc_scale * 2**truncation_bits).astype(np.float32)

    scale = onnx_ir.Value(
        name=scale_name, const_value=onnx_ir.tensor(trunc_scale, name=scale_name)
    )
    model.graph.register_initializer(scale)

    truncated_tensor = onnx_ir.Value(name=original_name)
    tensor.replace_all_uses_with(truncated_tensor, replace_graph_outputs=True)
    tensor.name = original_name + "_pretrunc"

    onnx_ir.Node(
        domain=_CUSTOM_OP_DOMAIN,
        op_type="Truncate",
        inputs=[tensor, scale],
        outputs=[truncated_tensor],
        name="Truncate_" + original_name,
        graph=model.graph,
    )


def _register_truncate_function(model: onnx_ir.Model):
    """
    Register the ``aimet.custom_ops::Truncate`` local function on ``model``.

    ``Truncate(input, scale) = floor(input / scale) * scale``, computed in the input's dtype
    """
    default_opset = model.opset_imports.get("", model.opset_imports.get("ai.onnx", 17))
    op = getattr(onnxscript, f"opset{default_opset}")

    @onnxscript.script(Opset(_CUSTOM_OP_DOMAIN, 1), default_opset=op)
    def Truncate(input, scale):  # pylint: disable=redefined-builtin
        scale = op.CastLike(scale, input)
        return op.Mul(op.Floor(op.Div(input, scale)), scale)

    function = onnx_ir.from_proto(Truncate.to_function_proto())
    model.functions[function.identifier()] = function
    model.opset_imports[_CUSTOM_OP_DOMAIN] = 1


def _broadcast_scale_to_output(scale: np.ndarray, op_type: str, output_rank: int):
    """
    Reshape a 1-D per-output-channel ``scale`` so it broadcasts against the op's output.

    For Gemm/MatMul: broadcast over last axis (keep 1-D)

    For Conv/ConvTranspose: broadcast over axis=1 (out_channels)
    """
    if op_type in ("Conv", "ConvTranspose") and output_rank >= 2:
        return scale.reshape((-1,) + (1,) * (output_rank - 2))
    return scale


def _get_affine_quantizers(sim, op):
    if not op.type in _LINEAR_OPS:
        return None, None, None

    # pylint: disable=protected-access
    input_qtzr = sim._get_enabled_quantizer(op.inputs[0].name)
    if not _is_int_quantizer(input_qtzr):
        input_qtzr = None

    weight_qtzr = sim._get_enabled_quantizer(op.inputs[1].name)
    if not _is_int_quantizer(weight_qtzr):
        weight_qtzr = None

    output_qtzr = sim._get_enabled_quantizer(op.outputs[0].name)
    if not _is_int_quantizer(output_qtzr):
        output_qtzr = None

    return input_qtzr, weight_qtzr, output_qtzr


def create_truncation_aware_session(
    sim: QuantizationSimModel, truncation_bits: int = 8
):
    # pylint: disable=protected-access
    matmul_ops = {
        op: sim._get_weight_and_bias(op)
        for op in sim.connected_graph.get_all_ops().values()
        if op.type in _LINEAR_OPS
    }

    model = onnx_ir.from_proto(sim.model.model)
    TopologicalSortPass()(model)
    output_ranks = _infer_output_ranks(model)
    value_map = onnx_ir.convenience.create_value_mapping(model.graph)

    for op, (weight, _) in matmul_ops.items():
        if weight is None:
            continue

        op_quantizers = _get_affine_quantizers(sim, op)
        if None in op_quantizers:
            continue

        input_qtzr, weight_qtzr, _ = op_quantizers

        # Blockwise weight scales aren't per-output-channel shaped
        if weight_qtzr._encoding_type() == EncodingType.PER_BLOCK:
            continue

        if weight_qtzr._encoding_type() == EncodingType.LPBQ:
            weight_scale = weight_qtzr._get_per_channel_scale()
        else:
            weight_scale = weight_qtzr._get_scale()

        input_scale = input_qtzr._get_scale()

        if weight_scale is None or input_scale is None:
            continue

        # int32 accumulator scale = input_scale * weight_scale.
        acc_scale = (input_scale * weight_scale).reshape(-1)

        output_name = op.outputs[0].name
        output_rank = output_ranks.get(output_name)
        if output_rank is None:
            continue
        _insert_output_truncation(
            model,
            value_map[output_name],
            _broadcast_scale_to_output(acc_scale, op.type, output_rank),
            truncation_bits,
        )

    _register_truncate_function(model)

    TopologicalSortPass()(model)

    session = aimet_onnx.utils.OrtInferenceSession(
        onnx_ir.to_proto(model),
        sim.providers,
        sim._ort_session_options,
        sim._path,
        sim._use_external_data,
    )

    return session
