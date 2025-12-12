# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from pytest import approx
import pytest
import numpy as np
from aimet_onnx._encoding import (
    AffineEncoding,
    LPBQEncoding,
    FloatEncoding,
    _float16,
    _bfloat16,
)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_affine_encoding_to_dict(dtype: str):
    unsigned, _ = dtype.split("int")

    e = AffineEncoding(
        scale=np.array(0.1, dtype=np.float32),
        offset=np.array(0, dtype=np.float32),
        dtype=dtype,
    )

    expected_scale = approx(0.1, rel=1e-6)
    if unsigned:
        expected_max = approx(25.5, rel=1e-6)
        expected_min = approx(0, rel=1e-6)
        expected_offset = 0
    else:
        expected_max = approx(12.7, rel=1e-6)
        expected_min = approx(-12.8, rel=1e-6)
        expected_offset = -128

    assert e.to_qnn_encoding_dict("0.6.1") == [
        {
            "max": expected_max,
            "min": expected_min,
            "scale": expected_scale,
            "offset": expected_offset,
            "bitwidth": 8,
            "dtype": "int",
            "is_symmetric": str(not unsigned),
        }
    ]
    assert e.to_qnn_encoding_dict("1.0.0") == {
        "dtype": "INT",
        "enc_type": "PER_TENSOR",
        "bw": 8,
        "is_sym": not unsigned,
        "scale": [expected_scale],
        "offset": [expected_offset],
    }
    assert e.to_qnn_encoding_dict("2.0.0") == {
        "output_dtype": dtype,
        "y_scale": expected_scale,
    }

    tf_encoding = e.to_TfEncoding()[0]
    assert tf_encoding.min == expected_min
    assert tf_encoding.max == expected_max
    assert tf_encoding.delta == expected_scale
    assert tf_encoding.offset == expected_offset
    assert tf_encoding.bw == 8


@pytest.mark.parametrize(
    "channel_axis, block_axis, block_size",
    [
        (None, None, None),
        (0, None, None),
        (0, 1, 4),
    ],
)
@pytest.mark.parametrize("zero_point_shift", [0.0, 0.5])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_affine_encoding_from_dict(
    dtype: str,
    zero_point_shift: float,
    channel_axis: int | None,
    block_axis: int | None,
    block_size: int | None,
):
    if channel_axis == block_axis == None:
        scale = np.array(0.01, dtype=np.float32)
        offset = np.array(0.0, dtype=np.int64)
    else:
        shape = (3, 2 * block_size) if block_axis is not None else (3,)
        numel = np.prod(shape)
        scale = np.arange(1, numel + 1, dtype=np.float32).reshape(shape) * 0.01
        offset = np.arange(numel, dtype=np.int64).reshape(shape)

    e = AffineEncoding(
        scale=scale,
        offset=offset + zero_point_shift,
        dtype=dtype,
        channel_axis=channel_axis,
        block_axis=block_axis,
        block_size=block_size,
    )

    assert e.to_unsigned() == e.to_signed().to_unsigned()
    assert e.to_signed() == e.to_unsigned().to_signed()

    if block_axis is None and zero_point_shift == 0.0:
        e2 = AffineEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("0.6.1"))
        assert AffineEncoding.is_equal(
            e.to_unsigned(),
            e2.to_unsigned(),
            # e2.channel_axis will be "auto" because 0.6.1 doesn't specify axis explicitly
            allow_auto_axis=not (channel_axis == block_axis == None),
        )

    e2 = AffineEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("1.0.0"))
    assert AffineEncoding.is_equal(
        e.to_unsigned(),
        e2.to_unsigned(),
        # e2.channel/block_axis will be "auto" because 1.0.0 doesn't specify axis explicitly
        allow_auto_axis=not (channel_axis == block_axis == None),
    )

    e2 = AffineEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("2.0.0"))
    assert AffineEncoding.is_equal(
        e.to_unsigned(),
        e2.to_unsigned(),
        # e2.channel_axis can be "auto" because
        # 2.0.0 doesn't specify channel_axis explicitly if block_axis is given
        allow_auto_axis=block_axis != None,
    )


@pytest.mark.parametrize("channel_axis, block_axis", [(0, 1), (1, 0)])
def test_lpbq_encoding_to_dict(channel_axis: int, block_axis: int):
    per_channel_float_scale = (
        np.arange(0.01, 0.11, 0.01, dtype=np.float64)
        .reshape(10, 1)
        .transpose((channel_axis, block_axis))
    )
    per_block_int_scale = np.random.randint(1, 16, (10, 10), dtype=np.int64)

    e = LPBQEncoding(
        per_channel_float_scale=per_channel_float_scale,
        per_block_int_scale=per_block_int_scale,
        dtype="int4",
        channel_axis=channel_axis,
        block_axis=block_axis,
        block_size=32,
    )

    assert e.to_qnn_encoding_dict("1.0.0") == {
        "dtype": "INT",
        "enc_type": "LPBQ",
        "compressed_bw": 4,
        "bw": 8,
        "block_size": 32,
        "scale": per_channel_float_scale.flatten().tolist(),
        "per_block_int_scale": per_block_int_scale.transpose((channel_axis, block_axis))
        .flatten()
        .tolist(),
        "offset": [-128] * 10,
        "is_sym": True,
    }
    assert e.to_qnn_encoding_dict("2.0.0") == {
        "output_dtype": "int4",
        "per_channel_float_scale": per_channel_float_scale.tolist(),
        "per_block_int_scale": per_block_int_scale.tolist(),
        "axis": block_axis,
        "block_size": 32,
    }

    # TODO (kyunggeu)
    # e2 = LPBQEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("1.0.0"))
    # assert LPBQEncoding.is_equal(e, e2, allow_auto_axis=True)

    e2 = LPBQEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("2.0.0"))
    assert LPBQEncoding.is_equal(e, e2, allow_auto_axis=True)


def test_float_encoding_to_dict():
    assert _float16.to_qnn_encoding_dict("0.6.1") == [
        {"bitwidth": 16, "dtype": "float"}
    ]
    assert _float16.to_qnn_encoding_dict("1.0.0") == {
        "dtype": "FLOAT",
        "bw": 16,
        "enc_type": "PER_TENSOR",
    }
    assert _float16.to_qnn_encoding_dict("2.0.0") == {}

    assert _float16 == FloatEncoding.from_qnn_encoding_dict(
        _float16.to_qnn_encoding_dict("0.6.1")
    )
    assert _float16 == FloatEncoding.from_qnn_encoding_dict(
        _float16.to_qnn_encoding_dict("1.0.0")
    )

    with pytest.raises(RuntimeError):
        _ = _bfloat16.to_qnn_encoding_dict("0.6.1")

    with pytest.raises(RuntimeError):
        _ = _bfloat16.to_qnn_encoding_dict("1.0.0")

    assert _bfloat16.to_qnn_encoding_dict("2.0.0") == {}
