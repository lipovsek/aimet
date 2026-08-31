# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest

from pathlib import Path
import random
import itertools
import torch
import numpy as np
import onnx
import onnxruntime as ort
import warnings
import aimet_torch
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization import DequantizedTensor
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize, FloatEncoding
from aimet_torch.quantization.float.quantizer import _fake_cast_to_ieee_float
from aimet_torch.quantization.float._finfo import _finfo, _float8_e4m3fn


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(999)
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture()
def x():
    """
    Returns [
        [-2., -1.99, -1.98, ..., -1.01],
        [-1., -0.99, -0.98, ..., -0.01],
        [ 0.,  0.01,  0.02, ...,  0.99],
        [ 1.,  1.01,  1.02, ...,  1.99],
    ]
    """
    return torch.arange(-200, 200).view(4, 100) / 100


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype, exponent_bits, mantissa_bits, finite, unsigned_zero",
    [
        (torch.float16, 5, 10, False, False),
        (torch.bfloat16, 8, 7, False, False),
        (torch.float8_e5m2, 5, 2, False, False),
        (torch.float8_e4m3fn, 4, 3, True, False),
        (torch.float8_e5m2fnuz, 5, 2, True, True),
        (torch.float8_e4m3fnuz, 4, 3, True, True),
    ],
)
def test_qdq_output_standard_dtypes(
    x, dtype, exponent_bits, mantissa_bits, finite, unsigned_zero
):
    """
    Given: Instantiated FloatQuantizeDequantize with a well-known dtype of pytorch
    When: Run forward
    Then: Output should be equal to downcasting and upcasting the input
    """
    float_qdq = FloatQuantizeDequantize(dtype=dtype)
    expected_output = x.to(dtype).float()
    assert torch.equal(float_qdq(x), expected_output)

    """
    Given: Instantiated two quantizers:
        - FloatQuantizeDequantize(dtype=dtype)
        - FloatQuantizeDequantize(exponent_bits, mantissa_bits)

        where exponent_bits and mantissa_bits corresponds to dtype
    When: Run forward
    Then: The two quantizers should produce same output
    """
    float_qdq_1 = FloatQuantizeDequantize(dtype=dtype)
    float_qdq_2 = FloatQuantizeDequantize(
        exponent_bits, mantissa_bits, finite, unsigned_zero
    )
    assert float_qdq_1._finfo == float_qdq_2._finfo
    assert (float_qdq_1.encoding_analyzer is None) == (float_qdq_1.bitwidth >= 16)
    assert (float_qdq_2.encoding_analyzer is None) == (float_qdq_2.bitwidth >= 16)

    float_qdq_out_1 = float_qdq_1(x)
    float_qdq_out_2 = float_qdq_2(x)
    assert torch.equal(float_qdq_out_1, float_qdq_out_2)
    assert isinstance(float_qdq_out_1, DequantizedTensor)
    assert isinstance(float_qdq_out_2, DequantizedTensor)
    assert (
        float_qdq_out_1.encoding._finfo
        == float_qdq_out_2.encoding._finfo
        == float_qdq_1._finfo
    )
    assert float_qdq_out_1.dequantize() is float_qdq_out_1
    assert float_qdq_out_2.dequantize() is float_qdq_out_2

    """
    When: Run compute_encodings() and forward again
    Then:
      1. The two quantizers should still produce same output
      2. If sub-16 floating point, compute_encodings should update its maxval
    """
    with float_qdq_1.compute_encodings(), float_qdq_2.compute_encodings():
        _ = float_qdq_1(x)
        _ = float_qdq_2(x)

    float_qdq_out_1_post_calib = float_qdq_1(x)
    float_qdq_out_2_post_calib = float_qdq_2(x)
    assert torch.equal(float_qdq_out_1, float_qdq_out_2)
    assert isinstance(float_qdq_out_1, DequantizedTensor)
    assert isinstance(float_qdq_out_2, DequantizedTensor)
    assert (
        float_qdq_out_1_post_calib.encoding._finfo
        == float_qdq_out_2_post_calib.encoding._finfo
        == float_qdq_1._finfo
    )
    assert float_qdq_out_1_post_calib.dequantize() is float_qdq_out_1_post_calib
    assert float_qdq_out_2_post_calib.dequantize() is float_qdq_out_2_post_calib

    if float_qdq_1.bitwidth < 16:
        assert not torch.isclose(float_qdq_out_1, float_qdq_out_1_post_calib).all()
        assert not torch.isclose(float_qdq_out_2, float_qdq_out_2_post_calib).all()
    else:
        assert torch.equal(float_qdq_out_1, float_qdq_out_1_post_calib)
        assert torch.equal(float_qdq_out_2, float_qdq_out_2_post_calib)

    assert float_qdq_out_1_post_calib.encoding.producer is float_qdq_1
    assert float_qdq_out_2_post_calib.encoding.producer is float_qdq_2


@pytest.mark.parametrize(
    "finite, unsigned_zero",
    [
        (True, True),
        (True, False),
        (False, True),
    ],
)
def test_special_floats_sanity(finite, unsigned_zero):
    ...
    """
    When: Instantiate non-builtin finite/unsigned_zero float qdq
    Then: Should throw runtime error
    """
    with pytest.raises(RuntimeError):
        _ = FloatQuantizeDequantize(3, 3, finite=finite, unsigned_zero=unsigned_zero)

    """
    Given: Start from a non-fininte, non-unsigned_zero float qdq
    When: Forcefully set fininte/unsigned_zero to True
    Then: Should throw runtime error
    """
    qdq = FloatQuantizeDequantize(3, 3, finite=False, unsigned_zero=False)
    qdq._finfo = _finfo(qdq.exponent_bits, qdq.mantissa_bits, finite, unsigned_zero)
    with pytest.raises(RuntimeError):
        _ = qdq(torch.randn(10))


@torch.no_grad()
def test_qdq_output_non_standard_dtype():
    """
    Given: Instantiated FloatQuantizeDequantize with a non-standard float dtype
    When: Run forward
    Then: Output should be equal to fake-casting the input to the non-standard float
    """
    #  float4_e2m1fn
    # |  in  | out  |
    # |------|------|
    # | -6.5 | -6.0 |
    # | -6.0 | -6.0 |
    # | -5.5 | -6.0 |
    # | -5.0 | -4.0 |
    # | -4.5 | -4.0 |
    # | -4.0 | -4.0 |
    # | -3.5 | -4.0 |
    # | -3.0 | -3.0 |
    # | -2.5 | -2.0 |
    # | -2.0 | -2.0 |
    # | -1.5 | -1.5 |
    # | -1.0 | -1.0 |
    # | -0.5 | -0.5 |
    # |  0.0 |  0.0 |
    # |  0.5 |  0.5 |
    # |  1.0 |  1.0 |
    # |  1.5 |  1.5 |
    # |  2.0 |  2.0 |
    # |  2.5 |  2.0 |
    # |  3.0 |  3.0 |
    # |  3.5 |  4.0 |
    # |  4.0 |  4.0 |
    # |  4.5 |  4.0 |
    # |  5.0 |  4.0 |
    # |  5.5 |  6.0 |
    # |  6.0 |  6.0 |
    # |  6.5 |  6.0 |
    x = torch.tensor(
        [-6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5]
        + [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
    )
    expected_output = torch.tensor(
        [-6, -6, -6, -4, -4, -4, -4, -3, -2, -2, -1.5, -1, -0.5]
        + [0, 0.5, 1, 1.5, 2, 2, 3, 4, 4, 4, 4, 6, 6, 6]
    )
    float4_e2m1fn_qdq = FloatQuantizeDequantize(
        exponent_bits=2,
        mantissa_bits=1,
        finite=True,
        unsigned_zero=False,
    )
    assert torch.equal(float4_e2m1fn_qdq(x), expected_output)
    assert torch.equal(
        expected_output,
        _fake_cast_to_ieee_float(x, float4_e2m1fn_qdq._finfo),
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
    ],
)
@pytest.mark.parametrize(
    "shape, block_size",
    [
        [(100,), None],
        [(10, 1), None],
        [(10, 2), (1, 50)],
        [(2, 2), (5, 50)],
    ],
)
def test_qdq_compute_encodings(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    block_size: tuple[int, ...] | None,
):
    """
    Given: Instantiated FloatQuantizeDequantize with a min-max encoding analyzer
    When: compute_encodings() and run forwad
    Then: Output should be equal to fake-casting the input
          with maximum representable value = observed maximum input
    """
    float8_tiny = torch.finfo(dtype).tiny
    float8_max = torch.finfo(dtype).max
    for x in [
        torch.arange(-2, 2, 0.004) * float8_max,
        torch.arange(-0.5, 0.5, 0.001) * float8_max,
    ]:
        x = x.view(10, 100)

        float8_qdq = FloatQuantizeDequantize(
            dtype=dtype, shape=shape, block_size=block_size
        )
        with float8_qdq.compute_encodings():
            _ = float8_qdq(x)

        output = float8_qdq(x)
        scale = float8_qdq.get_scale().unsqueeze(0)

        if block_size:
            B0, B1 = block_size
        else:
            B0 = x.shape[-2] // scale.shape[-2]
            B1 = x.shape[-1] // scale.shape[-1]

        for i, j in itertools.product(range(scale.shape[-2]), range(scale.shape[-1])):
            blk_input = x[i * B0 : (i + 1) * B0, j * B1 : (j + 1) * B1]
            blk_scale = scale[..., i, j]
            expected_blk_scale = blk_input.abs().amax() / float8_max
            assert torch.allclose(blk_scale, expected_blk_scale)

            blk_output = output[i * B0 : (i + 1) * B0, j * B1 : (j + 1) * B1]
            expected_blk_output = (blk_input / blk_scale).clamp(
                -float8_max, float8_max
            ).to(dtype).to(x.dtype) * blk_scale
            assert torch.allclose(blk_output, expected_blk_output, atol=float8_tiny)


def test_allow_overwrite(x):
    exponent_bits, mantissa_bits = 3, 4
    q = FloatQuantizeDequantize(exponent_bits, mantissa_bits, shape=(1, 100))
    with q.compute_encodings():
        q(x)
    q_max = q.maxval.detach().clone()

    """
    Given: allow_overwrite set to False
    When: Try to recompute encodings
    Then: Encoding does NOT get overwritten by compute_encodings
    """
    q.allow_overwrite(False)
    assert not q.is_overwrite_allowed("maxval")
    # Check deprecated _allow_overwrite flag for backwards compatibility
    assert not q._allow_overwrite

    with q.compute_encodings():
        q(x * 2)

    assert torch.equal(q_max, q.maxval)

    """
    Given: allow_overwrite set to True
    When: Try to recompute encodings
    Then: Encoding does NOT get overwritten by compute_encodings
    """
    q.allow_overwrite(True)
    assert q.is_overwrite_allowed("maxval")
    # Check deprecated _allow_overwrite flag for backwards compatibility
    assert q._allow_overwrite

    with q.compute_encodings():
        q(x * 2)

    assert torch.equal(q.maxval, q_max * 2)


def test_save_and_load_state_dict():
    float8_e5m2_qtzr = FloatQuantizeDequantize(dtype=torch.float8_e5m2)
    float8_e4m3fn_qtzr = FloatQuantizeDequantize(
        dtype=torch.float8_e4m3fn, shape=(20, 20), block_size=(5, 5)
    )

    dummy_input = torch.randn(100, 100)
    with float8_e5m2_qtzr.compute_encodings(), float8_e4m3fn_qtzr.compute_encodings():
        _ = float8_e5m2_qtzr(dummy_input)
        _ = float8_e4m3fn_qtzr(dummy_input)

    assert not torch.allclose(
        float8_e5m2_qtzr(dummy_input), float8_e4m3fn_qtzr(dummy_input)
    )

    float8_e5m2_qtzr.load_state_dict(float8_e4m3fn_qtzr.state_dict())
    assert float8_e5m2_qtzr._finfo == float8_e4m3fn_qtzr._finfo == _float8_e4m3fn
    assert float8_e5m2_qtzr.shape == float8_e4m3fn_qtzr.shape == (20, 20)
    assert float8_e5m2_qtzr.block_size == float8_e4m3fn_qtzr.block_size == (5, 5)
    assert torch.equal(float8_e5m2_qtzr(dummy_input), float8_e4m3fn_qtzr(dummy_input))


def test_extreme_values_warning():
    extreme_val = torch.finfo(torch.float16).max
    dummy_input = torch.arange(start=0, end=extreme_val, dtype=torch.float16)
    encoding_shape = (1,)
    qdq = FloatQuantizeDequantize(
        dtype=torch.float16, encoding_analyzer=MinMaxEncodingAnalyzer(encoding_shape)
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with qdq.compute_encodings():
            qdq(dummy_input)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Extreme values" in str(w[-1].message)


@torch.no_grad()
@pytest.mark.parametrize("dynamo", [True, False])
@pytest.mark.parametrize(
    "shape, block_size",
    [
        [(1, 10), None],
        [(10, 1), (1, 10)],
        [(10, 2), (1, 5)],
    ],
)
@pytest.mark.parametrize(
    "dtype, maxval",
    [
        (torch.float16, None),
        (torch.float8_e5m2, None),
        (torch.float8_e5m2, 16.0),
    ],
)
def test_onnx_export(
    dtype: torch.dtype,
    maxval: float | None,
    shape: tuple[int, ...],
    block_size: tuple[int, ...] | None,
    dynamo: bool,
    tmp_path: Path,
):
    """
    When: torch.onnx.export a quantizer
    Then: export shouldn't throw error
    """
    qdq = FloatQuantizeDequantize(dtype=dtype, shape=shape, block_size=block_size)
    x = torch.randn(10, 10)

    if maxval is not None:
        qdq.maxval = torch.full(shape, maxval)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(x)
            self.qdq = qdq

        def forward(self, x):
            return x + self.qdq(self.x)

    model = Model()
    torch.onnx.export(
        model,
        (x,),
        tmp_path / "float_qdq_pre.onnx",
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
    )
    onnx_qdq_model = onnx.load(tmp_path / "float_qdq_pre.onnx")
    onnx.checker.check_model(onnx_qdq_model)

    aimet_torch.onnx.export(
        model,
        (x,),
        tmp_path / "float_qdq.onnx",
        dynamo=dynamo,
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
    )
    onnx_qdq_model = onnx.load(tmp_path / "float_qdq.onnx")
    onnx.checker.check_model(onnx_qdq_model)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_qdq_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": x.detach().numpy()})

    if dtype == torch.float8_e5m2:
        expected_out = x + qdq(x)
    else:
        # float16 quantizer won't be actually exported to onnx QDQ.
        expected_out = x + x

    assert torch.allclose(torch.from_numpy(out), expected_out)


def test_float_encoding_to():
    """
    When: FloatEncoding with scale=None
    Then: Throw ValueError
    """
    with pytest.raises(ValueError):
        encoding = FloatEncoding(
            exponent_bits=5,
            mantissa_bits=10,
            finite=False,
            unsigned_zero=False,
            scale=None,
            block_size=(5, 10),
        )

    """
    Given: FloatEncoding with scale!=None
    """
    encoding = FloatEncoding(
        exponent_bits=5,
        mantissa_bits=10,
        finite=False,
        unsigned_zero=False,
        scale=torch.tensor(0.1),
        block_size=(5, 10),
    )
    """
    When: Call .to() with same dtype and device
    Then: Should return identical object
    """
    new_encoding = encoding.to(device="cpu", dtype=torch.float32)
    assert new_encoding is encoding

    """
    When: Call .to() with new dtype and device
    Then: 1. New encoding object should be in proper dtype and device
          2. Old encoding object should not be affected
    """
    new_encoding = encoding.to(device="cpu", dtype=torch.float16)
    assert new_encoding.maxval.device == torch.device("cpu")
    assert new_encoding.maxval.dtype == torch.float16
    assert new_encoding.block_size == encoding.block_size

    assert encoding.maxval.device == torch.device("cpu")
    assert encoding.maxval.dtype == torch.float32


def test_default_args():
    float16_qdq = FloatQuantizeDequantize(exponent_bits=5, mantissa_bits=10)
    assert float16_qdq.is_float16()

    bfloat16_qdq = FloatQuantizeDequantize(exponent_bits=8, mantissa_bits=7)
    assert bfloat16_qdq.is_bfloat16()


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_qdq_ignore_boolean_and_integers(dtype):
    """
    When: Pass boolean or integer tensor as input to FloatQuantizeDequantize
    Then: Output should be same as input, with dtype preserved.
    """
    x = (torch.arange(10) % 2).to(dtype)
    float_qdq = FloatQuantizeDequantize(dtype=torch.float8_e5m2)
    out = float_qdq(x)
    assert torch.equal(out, x)
    assert out.dtype == x.dtype


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "shape, block_size",
    [
        ((), None),  # per-tensor
        ((10, 1), None),  # per-channel with axis=0
        ((10, 2), (-1, -1)),  # per-block with channel_axis=0, block_axis=1
    ],
)
def test_fullgraph_compile(shape, block_size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available")

    qdq = FloatQuantizeDequantize(
        dtype=torch.float8_e5m2, shape=shape, block_size=block_size
    ).to(device)

    with qdq.compute_encodings():
        qdq(torch.randn(10, 10, device=device))

    qdq = torch.compile(qdq, fullgraph=True)

    x = torch.randn(10, 10, device=device)
    _ = qdq(x)

    # Re-run with different shape to trigger recompilation
    x = torch.randn(2, 10, 10, device=device)
    _ = qdq(x)


@pytest.fixture(autouse=True)
def clear_torch_compile_cache():
    yield
    torch.compiler.reset()
