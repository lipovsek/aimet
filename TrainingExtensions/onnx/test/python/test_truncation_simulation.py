# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest
from onnx import helper, numpy_helper, TensorProto

from aimet_onnx import QuantizationSimModel, int4, int16
from aimet_onnx.common.utils import compute_psnr
from aimet_onnx.quantsim import set_lpbq_for_params
from aimet_onnx.utils import numpy_to_TfEncoding
from aimet_onnx.experimental._truncation_aware import create_truncation_aware_session

_CHANNELS = 4


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def _mask_low_bits(accumulator_int32, truncation_bits):
    """Zero the low ``truncation_bits`` of a two's-complement int32 (a floor toward -inf)."""
    return accumulator_int32 & ~np.int64(2**truncation_bits - 1)


def _input_shape(op_type, channels):
    return (1, channels, 2, 2) if op_type == "Conv" else (3, channels)


def _build_model(op_type, weight, bias):
    """A model with a single Conv/Gemm/MatMul op, optionally including a bias add."""
    channels = weight.shape[0]
    inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, _input_shape(op_type, channels)
        )
    ]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]
    initializers = [numpy_helper.from_array(weight, "weight")]

    if op_type == "MatMul":
        # MatMul takes its bias through a separate Add, which quantsim fuses into a Gemm.
        matmul_output = "matmul_out" if bias is not None else "output"
        nodes = [helper.make_node("MatMul", ["input", "weight"], [matmul_output])]
        if bias is not None:
            nodes.append(helper.make_node("Add", [matmul_output, "bias"], ["output"]))
    else:
        op_inputs = ["input", "weight"] + (["bias"] if bias is not None else [])
        nodes = [helper.make_node(op_type, op_inputs, ["output"])]

    if bias is not None:
        initializers.append(numpy_helper.from_array(bias, "bias"))

    graph = helper.make_graph(nodes, f"{op_type}_model", inputs, outputs, initializers)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
    )


def _build_calibrated_sim(op_type, weight, with_bias, calibration_input):
    bias = np.zeros(_CHANNELS, np.float32) if with_bias else None
    sim = QuantizationSimModel(
        _build_model(op_type, weight, bias),
        param_type=int4,
        activation_type=int16,
    )
    sim.compute_encodings([{"input": calibration_input}])
    return sim


def _build_calibrated_sim_with_identity_weight(op_type, with_bias, calibration_input):
    identity = np.eye(_CHANNELS, dtype=np.float32)
    if op_type == "Conv":
        identity = identity.reshape(_CHANNELS, _CHANNELS, 1, 1)
    # Calibrate before overriding the weight encodings so activation ranges are realistic.
    sim = _build_calibrated_sim(op_type, identity, with_bias, calibration_input)

    # Set weight scale to 1 (int32 accumulator becomes exactly int16 input)
    quantizer = sim.qc_quantize_op_dict["weight"]
    scale = np.ones(_CHANNELS, np.float32)
    offset = np.full(_CHANNELS, -(2 ** (int4.bits - 1)), np.float32)
    quantizer.load_encodings(numpy_to_TfEncoding(scale, offset, int4))
    return sim


@pytest.mark.parametrize("op_type", ("Conv", "Gemm", "MatMul"))
class TestTruncationSimulation:
    @pytest.mark.parametrize("truncation_bits", (4, 8))
    def test_truncation_masks_low_bits_of_accumulator(self, op_type, truncation_bits):
        """Truncation floors the int32 accumulator onto a `truncation_bits`-coarser grid."""
        sample = np.random.randn(*_input_shape(op_type, _CHANNELS)).astype(np.float32)
        sim = _build_calibrated_sim_with_identity_weight(
            op_type, with_bias=False, calibration_input=sample
        )
        input_quantizer = sim.qc_quantize_op_dict["input"]
        output_quantizer = sim.qc_quantize_op_dict["output"]

        session = create_truncation_aware_session(sim, truncation_bits=truncation_bits)
        output = session.run(None, {"input": sample})[0]

        # The weight scale is 1, so acc_scale == input_scale and the accumulator in int32
        # units is round(quantized_input / input_scale). Zero its low bits and dequantize.
        acc_scale = input_quantizer._get_scale()
        quantized_input = input_quantizer.quantize_dequantize(sample)
        accumulator = np.round(quantized_input / acc_scale).astype(np.int64)
        truncated = _mask_low_bits(accumulator, truncation_bits) * acc_scale
        expected = output_quantizer.quantize_dequantize(truncated.astype(np.float32))
        output_scale = output_quantizer._get_scale()
        assert np.allclose(output, expected, atol=output_scale)

    @pytest.mark.parametrize("truncation_bits", (4, 8))
    def test_truncation_masks_low_bits_of_bias(self, op_type, truncation_bits):
        """With a zero input, the output is the int32 bias with its low bits masked off."""
        shape = _input_shape(op_type, _CHANNELS)
        sample = np.random.randn(*shape).astype(np.float32)

        weight_shape = (
            (_CHANNELS, _CHANNELS, 1, 1)
            if op_type == "Conv"
            else (_CHANNELS, _CHANNELS)
        )
        weight = np.random.randn(*weight_shape).astype(np.float32)
        sim = _build_calibrated_sim(
            op_type, weight, with_bias=True, calibration_input=sample
        )
        output_quantizer = sim.qc_quantize_op_dict["output"]

        input_scale = sim.qc_quantize_op_dict["input"]._get_scale()
        weight_scale = sim.qc_quantize_op_dict["weight"]._get_scale().reshape(-1)
        acc_scale = input_scale * weight_scale
        encoding = output_quantizer.get_encodings()[0]

        # Bias represented in int32 accumulator units
        bias_int32 = np.random.randint(
            int(encoding.min / acc_scale.max()),
            int(encoding.max / acc_scale.max()),
            _CHANNELS,
            dtype=np.int64,
        )
        bias = sim.model.get_initializer("bias")
        bias.CopyFrom(
            numpy_helper.from_array((bias_int32 * acc_scale).astype(np.float32), "bias")
        )

        session = create_truncation_aware_session(sim, truncation_bits=truncation_bits)

        # Output with zero-value input should be exactly truncated bias through output QDQ
        output = session.run(None, {"input": np.zeros_like(sample)})[0]
        truncated = _mask_low_bits(bias_int32, truncation_bits) * acc_scale
        if op_type == "Conv":
            truncated = truncated.reshape(_CHANNELS, 1, 1)
        expected = output_quantizer.quantize_dequantize(
            np.broadcast_to(truncated, output.shape).astype(np.float32)
        )
        output_scale = output_quantizer._get_scale()
        assert np.allclose(output, expected, atol=output_scale)


@pytest.mark.parametrize("lpbq", (True, False))
def test_truncation_error_matches_integer_reference(lpbq):
    """
    On a random matmul, truncation simulation matches an explicit int32 accumulator
    reference better than vanilla quantsim.
    """
    truncation_bits = 8
    channels = 10
    input_tensor = np.random.randn(3, channels).astype(np.float32)
    weight_tensor = np.random.randn(channels, channels).astype(np.float32)
    bias_tensor = np.random.randn(channels).astype(np.float32)

    sim = QuantizationSimModel(
        _build_model("MatMul", weight_tensor, bias_tensor),
        param_type=int4,
        activation_type=int16,
    )
    if lpbq:
        # Decompress the int4 weights to an int8 grouped-blockwise representation.
        set_lpbq_for_params(
            sim,
            bitwidth=4,
            block_size=channels // 2,
            op_types={"MatMul", "Gemm"},
        )
    sim.compute_encodings([{"input": input_tensor}])

    input_quantizer = sim.qc_quantize_op_dict["input"]
    weight_quantizer = sim.qc_quantize_op_dict["weight"]
    output_quantizer = sim.qc_quantize_op_dict["output"]

    input_scale = input_quantizer._get_scale()
    # LPBQ weights are quantized on the finer per-channel int8 grid.
    if lpbq:
        scale_encoding = weight_quantizer._scale_encoding_dict()
        weight_scale = scale_encoding["x_scale"] if scale_encoding else None
    else:
        weight_scale = weight_quantizer._get_scale().reshape(-1)
    # int32 accumulator scale = input_scale * weight_scale, per output channel.
    acc_scale = input_scale * weight_scale

    # Perform basic integer matmul ignoring offsets (not considered in truncation currently)
    input_int = np.round(
        input_quantizer.quantize_dequantize(input_tensor) / input_scale
    )
    weight_int = np.round(
        weight_quantizer.quantize_dequantize(weight_tensor) / weight_scale
    )

    bias_int32 = np.round(bias_tensor / acc_scale).astype(np.int64)
    accumulator = (input_int @ weight_int).astype(np.int64) + bias_int32
    truncated = _mask_low_bits(accumulator, truncation_bits) * acc_scale
    expected = output_quantizer.quantize_dequantize(truncated.astype(np.float32))

    truncation_session = create_truncation_aware_session(
        sim, truncation_bits=truncation_bits
    )
    truncated_output = truncation_session.run(None, {"input": input_tensor})[0]
    plain_output = sim.session.run(None, {"input": input_tensor})[0]

    # Allow for 1 truncation step difference only for elements on int32 border line
    on_grid_line = accumulator % 2**truncation_bits == 0
    atol = output_quantizer._get_scale() + np.where(
        on_grid_line, acc_scale * 2**truncation_bits, 0.0
    )
    assert np.all(np.abs(expected - truncated_output) <= atol)

    # The truncation-aware session tracks the integer reference better than plain sim.
    assert compute_psnr(expected, truncated_output) > compute_psnr(
        expected, plain_output
    )
