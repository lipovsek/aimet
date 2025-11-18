# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.common.onnx._utils import _is_grid_preserving_op
from aimet_onnx.qc_quantize_op import QcQuantizeOp
import logging

logger = logging.getLogger(__name__)


def _get_quantizer_no_split_slice(
    quantsim_model: QuantSimOnnx, tensor_name: str
) -> QcQuantizeOp:
    """
    Returns closest enabled quantizer to tensor traversing upwards only through invariant ops and no Split/Slice

    :param tensor_name: Name of tensor for which to find quantizer
    """
    quantizer = quantsim_model.qc_quantize_op_dict.get(tensor_name, None)
    if quantizer and quantizer.enabled:
        return quantizer

    prod_dict = quantsim_model.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)

    if product == None:
        if tensor_name.endswith(("_updated", "_qdq")):
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Input tensor_name must be the name of a tensor in the original (unquantized) graph"
            )
        else:
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. Tensor name does not exist in the graph"
            )

    producer = product.producer

    if producer == None:
        return None

    if (
        not (_is_grid_preserving_op(producer.type))
        or producer.type == "Slice"
        or producer.type == "Split"
        or producer.type == "SplitToSequence"
    ):
        return None

    if len(producer.inputs) == 0:
        return None

    upstream_tensor = producer.inputs[0]
    return _get_quantizer_no_split_slice(quantsim_model, upstream_tensor.name)


def _set_matmul_second_input_to_8b(quantsim_model: QuantSimOnnx):
    cg = quantsim_model.connected_graph

    for op in reversed(cg.ordered_ops):
        if op.type != "MatMul":
            continue

        upper_quantizer = quantsim_model._get_enabled_quantizer(op.inputs[1].name)  # pylint: disable=protected-access

        enabled_quantizer = _get_quantizer_no_split_slice(
            quantsim_model, op.inputs[1].name
        )

        if enabled_quantizer and enabled_quantizer.bitwidth <= 8:
            continue
        elif enabled_quantizer:
            enabled_quantizer.set_bitwidth(8)
            enabled_quantizer.use_symmetric_encodings = True
        elif upper_quantizer:
            if op.inputs[1].name in quantsim_model.qc_quantize_op_dict:
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True
            else:
                quantsim_model._insert_quantizer(op.inputs[1].name, is_param=False)  # pylint: disable=protected-access
                quantsim_model._rebuild_session()  # pylint: disable=protected-access
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True


def _tie_quantizers_for_kv_cache(
    quantsim_model: QuantSimOnnx, kv_io_map: dict[str, str]
) -> None:
    quantizer_mapping = dict()

    for input_name, output_name in kv_io_map.items():
        quantizer = quantsim_model._get_enabled_quantizer(output_name)  # pylint: disable=protected-access
        if quantizer:
            quantizer_mapping[input_name] = quantizer
        else:
            logger.warning(
                "Warning: No valid quantizer found for output %s", output_name
            )

    quantsim_model.set_quantizers(quantizer_mapping)


def _set_lm_head_to_8b(quantsim_model: QuantSimOnnx, lm_head_tensor_name: str):
    quantizer = quantsim_model.qc_quantize_op_dict.get(lm_head_tensor_name, None)
    if quantizer == None:
        raise KeyError(
            f"Could not find quantizer for LM head tensor: {lm_head_tensor_name}"
        )
    quantizer.set_bitwidth(8)
    quantizer._enable_blockwise_quantization(0)  # pylint: disable=protected-access
    quantizer.enable_per_channel_quantization()


def _set_tensor_to_8_bit_symmetric(quantsim_model: QuantSimOnnx, tensor_name: str):
    quantizer = quantsim_model._get_enabled_quantizer(tensor_name)  # pylint: disable=protected-access
    if quantizer:
        quantizer.set_bitwidth(8)
        quantizer.use_symmetric_encodings = True
    else:
        logger.warning("Warning: No valid quantizer found for output %s", tensor_name)


def _set_tensors_to_output_8b_sym(quantsim_model: QuantSimOnnx, out_tensors: list[str]):
    for out_tensor in out_tensors:
        _set_tensor_to_8_bit_symmetric(quantsim_model, out_tensor)


def _apply_int8_kv_cache_tying_and_lm_head(
    sim: QuantSimOnnx, kv_io_map: dict[str, str], lm_head_tensor_name: str
):
    sim._tie_quantizers_for_op_types(["Concat"])  # pylint: disable=protected-access
    sim._rebuild_session()  # pylint: disable=protected-access

    # Setting kv_cache and some other layers to 8-bit
    kv_io_list = list(kv_io_map.keys()) + list(kv_io_map.values())
    _set_tensors_to_output_8b_sym(sim, kv_io_list)

    # Setting the LM head weights to 8-bit.
    _set_lm_head_to_8b(
        sim,
        lm_head_tensor_name,
    )

    # Tie kv_cache
    _tie_quantizers_for_kv_cache(sim, kv_io_map)

    # Setting Matmul second input to 8b
    _set_matmul_second_input_to_8b(sim)

    return sim
