# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import FrozenInstanceError

from pytest import approx
import pytest
import numpy as np
from aimet_onnx._encoding import (
    AffineEncoding,
    LPBQEncoding,
    FloatEncoding,
    _float16,
    _bfloat16,
    _float8e4m3fn,
    _float8e5m2,
)
from aimet_onnx.common import libpymo
from aimet_onnx.common.defs import qtype


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
        ("auto", None, None),
        (0, 1, 4),
        ("auto", "auto", 4),
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
        assert e.to_qnn_encoding_dict("0.6.1") == [
            {
                name: approx(value, rel=1e-6) if isinstance(value, float) else value
                for name, value in enc.items()
            }
            for enc in e2.to_qnn_encoding_dict("0.6.1")
        ]

    e2 = AffineEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("1.0.0"))
    assert AffineEncoding.is_equal(
        e.to_unsigned(),
        e2.to_unsigned(),
        # e2.channel/block_axis will be "auto" because 1.0.0 doesn't specify axis explicitly
        allow_auto_axis=not (channel_axis == block_axis == None),
    )
    assert e.to_qnn_encoding_dict("1.0.0") == e2.to_qnn_encoding_dict("1.0.0")

    if "auto" in (channel_axis, block_axis):
        with pytest.raises(RuntimeError):
            e2 = AffineEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("2.0.0"))
    else:
        e2 = AffineEncoding.from_qnn_encoding_dict(e.to_qnn_encoding_dict("2.0.0"))
        assert AffineEncoding.is_equal(
            e.to_unsigned(),
            e2.to_unsigned(),
            # e2.channel_axis can be "auto" because
            # 2.0.0 doesn't specify channel_axis explicitly if block_axis is given
            allow_auto_axis=isinstance(block_axis, int),
        )
        assert e.to_qnn_encoding_dict("2.0.0") == e2.to_qnn_encoding_dict("2.0.0")


@pytest.mark.parametrize("compressed_bw, decompressed_bw", [(4, 8), (2, 8)])
@pytest.mark.parametrize("channel_axis, block_axis", [(0, 1), (1, 0)])
def test_lpbq_encoding_to_dict(
    channel_axis: int, block_axis: int, compressed_bw: int, decompressed_bw: int
):
    per_channel_float_scale = (
        np.arange(0.01, 0.11, 0.01, dtype=np.float64)
        .reshape(10, 1)
        .transpose((channel_axis, block_axis))
    )
    per_block_int_scale = np.random.randint(1, 16, (10, 10), dtype=np.int64)

    e = LPBQEncoding(
        per_channel_float_scale=per_channel_float_scale,
        per_block_int_scale=per_block_int_scale,
        dtype=f"int{compressed_bw}",
        channel_axis=channel_axis,
        block_axis=block_axis,
        block_size=32,
        decompressed_dtype=f"int{decompressed_bw}",
    )

    assert e.to_qnn_encoding_dict("1.0.0") == {
        "dtype": "INT",
        "enc_type": "LPBQ",
        "compressed_bw": compressed_bw,
        "bw": decompressed_bw,
        "block_size": 32,
        "scale": per_channel_float_scale.flatten().tolist(),
        "per_block_int_scale": per_block_int_scale.transpose((channel_axis, block_axis))
        .flatten()
        .tolist(),
        "offset": [-128] * 10,
        "is_sym": True,
    }

    # 1.0.0 encoding can't be fully reconstructed unless
    # input_shape, channel_axis, block_axis are provided
    e2 = LPBQEncoding.from_qnn_encoding_dict(
        e.to_qnn_encoding_dict("1.0.0"),
        input_shape=(10, 320) if channel_axis == 0 else (320, 10),
        default_channel_axis=channel_axis,
        default_block_axis=block_axis,
    )
    assert LPBQEncoding.is_equal(e, e2, allow_auto_axis=True)

    if decompressed_bw != compressed_bw * 2:
        with pytest.raises(RuntimeError):
            _ = e.to_qnn_encoding_dict("2.0.0")
        return

    assert e.to_qnn_encoding_dict("2.0.0") == {
        "output_dtype": f"int{compressed_bw}",
        "per_channel_float_scale": per_channel_float_scale.tolist(),
        "per_block_int_scale": per_block_int_scale.tolist(),
        "axis": block_axis,
        "block_size": 32,
    }

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


_FLOAT8_FORMATS = {"float8e4m3fn": _float8e4m3fn, "float8e5m2": _float8e5m2}


def _scaled(dtype: str, scale, **kwargs) -> FloatEncoding:
    """Build a scaled FloatEncoding for the named float8 format."""
    fmt = _FLOAT8_FORMATS[dtype]
    return FloatEncoding(
        exponent_bits=fmt.exponent_bits,
        mantissa_bits=fmt.mantissa_bits,
        finite=fmt.finite,
        unsigned_zero=fmt.unsigned_zero,
        scale=np.array(scale, dtype=np.float32),
        **kwargs,
    )


@pytest.mark.parametrize(
    "dtype, expected_max",
    [("float8e4m3fn", 448.0), ("float8e5m2", 57344.0)],
)
def test_float8_max_representable_matches_cpp(dtype: str, expected_max: float):
    """
    The common Python helper must agree with the C++ QuantizationType::Float derivation,
    which is the value the kernel actually uses.
    """
    precision = qtype.from_string(dtype)
    assert _FLOAT8_FORMATS[dtype].max_representable == expected_max

    quantizer = libpymo.BlockTensorQuantizer.createFloat(
        [1],
        precision.exponent_bits,
        precision.mantissa_bits,
        precision.finite,
        precision.unsigned_zero,
    )
    assert quantizer.getFloatSpec()["max_value"] == expected_max


@pytest.mark.parametrize("dtype", ["float8e4m3fn", "float8e5m2"])
def test_float8_encoding_dtype_and_bitwidth(dtype: str):
    """The format fields must round-trip to the ONNX dtype name."""
    e = _scaled(dtype, 0.5)
    assert e.dtype == dtype
    assert e.bitwidth == 8
    assert e.is_scaled


@pytest.mark.parametrize("dtype", ["float8e4m3fn", "float8e5m2"])
def test_float8_encoding_per_tensor(dtype: str):
    e = _scaled(dtype, 0.5)

    assert e.to_qnn_encoding_dict("2.0.0") == {
        "output_dtype": dtype,
        "y_scale": approx(0.5, rel=1e-6),
    }

    # The 0.6.1/1.0.0 QNN schemas cannot describe a scaled float grid.
    for version in ("0.6.1", "1.0.0"):
        with pytest.raises(RuntimeError):
            _ = e.to_qnn_encoding_dict(version)


def test_float8_encoding_per_channel():
    e = _scaled("float8e4m3fn", [[0.5], [0.25]], channel_axis=0)

    assert e.to_qnn_encoding_dict("2.0.0") == {
        "output_dtype": "float8e4m3fn",
        "y_scale": [approx(0.5, rel=1e-6), approx(0.25, rel=1e-6)],
        "axis": 0,
    }


def test_float8_encoding_blockwise():
    """
    Blockwise keeps the scale multi-dimensional and emits block_size, matching
    AffineEncoding._to_2_0_0. Flattening it would misdescribe the grid.
    """
    scale = np.arange(1, 7, dtype=np.float32).reshape(3, 2) / 8
    e = _scaled("float8e4m3fn", scale, channel_axis=0, block_axis=1, block_size=4)

    d = e.to_qnn_encoding_dict("2.0.0")
    assert d["output_dtype"] == "float8e4m3fn"
    assert d["axis"] == 1
    assert d["block_size"] == 4
    assert np.array(d["y_scale"]).shape == (3, 2)
    assert np.allclose(np.array(d["y_scale"]), scale)


def test_float8_encoding_blockwise_negative_axes_round_trip():
    scale = np.arange(1, 7, dtype=np.float32).reshape(3, 2) / 8
    encoding = _scaled(
        "float8e4m3fn",
        scale,
        channel_axis=-2,
        block_axis=-1,
        block_size=4,
    )

    encoding_dict = encoding.to_qnn_encoding_dict("2.0.0")
    assert encoding_dict["axis"] == -1
    loaded = FloatEncoding.from_qnn_encoding_dict(
        encoding_dict, default_channel_axis=-2
    )
    assert loaded == encoding


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_axis": "auto"},
        {"channel_axis": 0, "block_axis": "auto", "block_size": 4},
    ],
)
def test_float8_encoding_rejects_auto_axis(kwargs):
    """Mirrors AffineEncoding: 'auto' cannot be resolved at export time."""
    e = _scaled("float8e4m3fn", [[0.5], [0.25]], **kwargs)
    with pytest.raises(RuntimeError, match="auto"):
        _ = e.to_qnn_encoding_dict("2.0.0")


@pytest.mark.parametrize(
    "dtype, expected_max", [("float8e4m3fn", 448.0), ("float8e5m2", 57344.0)]
)
def test_float8_encoding_to_TfEncoding(dtype: str, expected_max: float):
    """min/max must describe the float8 range, not an integer-derived grid."""
    scale = 0.5
    (tf_encoding,) = _scaled(dtype, scale).to_TfEncoding()

    assert tf_encoding.delta == approx(scale, rel=1e-6)
    assert tf_encoding.offset == 0
    assert tf_encoding.bw == 8
    assert tf_encoding.max == approx(expected_max * scale, rel=1e-6)
    assert tf_encoding.min == approx(-expected_max * scale, rel=1e-6)


def test_plain_cast_float_has_no_grid():
    """fp16/bf16 carry no scale, so there is no TfEncoding to produce."""
    assert not _float16.is_scaled
    assert not _bfloat16.is_scaled
    with pytest.raises(RuntimeError, match="plain cast"):
        _ = _float16.to_TfEncoding()


@pytest.mark.parametrize(
    "encoding, fields",
    [
        (_float16, (5, 10, False, False)),
        (_bfloat16, (8, 7, False, False)),
    ],
)
def test_plain_cast_float_preserves_frozen_dataclass_behavior(encoding, fields):
    """The class merge must not change the pre-existing fp16/bf16 value objects."""
    exponent_bits, mantissa_bits, finite, unsigned_zero = fields
    assert repr(encoding) == (
        f"FloatEncoding(exponent_bits={exponent_bits}, "
        f"mantissa_bits={mantissa_bits}, finite={finite}, "
        f"unsigned_zero={unsigned_zero})"
    )
    assert hash(encoding) == hash(fields)
    with pytest.raises(FrozenInstanceError):
        encoding.exponent_bits = exponent_bits + 1
    with pytest.raises(FrozenInstanceError):
        del encoding.exponent_bits


@pytest.mark.parametrize(
    "field, value",
    [("exponent_bits", 5), ("scale", np.array(0.25)), ("channel_axis", 0)],
)
def test_scaled_float_encoding_is_frozen(field, value):
    """Scaled encodings are value objects too; the format and grid can't be rewritten."""
    encoding = _scaled("float8e4m3fn", 0.5)
    with pytest.raises(FrozenInstanceError):
        setattr(encoding, field, value)


def test_scaled_float_encoding_hash_is_stable_when_scale_changes():
    """Hashing mutable ndarray bytes would corrupt sets after in-place scale updates."""
    encoding = _scaled("float8e4m3fn", 0.5)
    encodings = {encoding}
    original_hash = hash(encoding)

    encoding.scale[...] = 0.25

    assert hash(encoding) == original_hash
    assert encoding in encodings


def test_float8_encoding_allclose_and_auto_axis_equality():
    """Keep the comparison helpers formerly inherited from AffineEncoding."""
    expected = _scaled(
        "float8e4m3fn",
        [[0.5, 0.25]],
        channel_axis=0,
        block_axis=1,
        block_size=4,
    )
    loaded = _scaled(
        "float8e4m3fn",
        [0.500001, 0.250001],
        channel_axis="auto",
        block_axis="auto",
        block_size=4,
    )

    assert not expected.is_equal(loaded)
    assert expected.allclose(loaded, atol=2e-6, allow_auto_axis=True)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"scale": [[0.5, 0.25]], "block_axis": 1, "block_size": 4}, "channel_axis"),
        (
            {"scale": [[0.5, 0.25]], "channel_axis": 0, "block_axis": 1},
            "both specified",
        ),
        (
            {
                "scale": [0.5, 0.25],
                "channel_axis": 0,
                "block_axis": -1,
                "block_size": 4,
            },
            "block_axis",
        ),
    ],
)
def test_float8_encoding_rejects_invalid_block_granularity(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _scaled("float8e4m3fn", **kwargs)


@pytest.mark.parametrize("dtype", ["float8e4m3fn", "float8e5m2"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"channel_axis": 0},
        {"channel_axis": 0, "block_axis": 1, "block_size": 4},
    ],
    ids=["per_tensor", "per_channel", "blockwise"],
)
def test_float8_encoding_2_0_0_round_trip(dtype: str, kwargs: dict):
    """A 2.0.0 dict must parse back into an equal encoding, at every granularity."""
    if "block_axis" in kwargs:
        scale = np.arange(1, 7, dtype=np.float32).reshape(3, 2) / 8
    elif "channel_axis" in kwargs:
        scale = np.array([[0.5], [0.25]], dtype=np.float32)
    else:
        scale = np.array(0.5, dtype=np.float32)

    e = _scaled(dtype, scale, **kwargs)
    d = e.to_qnn_encoding_dict("2.0.0")
    # The 2.0.0 dict carries only one axis, so a blockwise dict cannot express the
    # channel axis; it is recovered from default_channel_axis, as for AffineEncoding.
    loaded = FloatEncoding.from_qnn_encoding_dict(
        d, default_channel_axis=e.channel_axis if e.block_axis is not None else None
    )

    assert loaded.dtype == e.dtype
    assert loaded.channel_axis == e.channel_axis
    assert loaded.block_axis == e.block_axis
    assert loaded.block_size == e.block_size
    assert np.allclose(
        np.asarray(loaded.scale).flatten(), np.asarray(e.scale).flatten()
    )


def test_float8_encoding_equality_distinguishes_formats():
    """(float, 8) cannot tell e4m3fn from e5m2, so equality must use the format fields."""
    assert _scaled("float8e4m3fn", 0.5) != _scaled("float8e5m2", 0.5)
    assert _scaled("float8e4m3fn", 0.5) != _scaled("float8e4m3fn", 0.25)
    assert _scaled("float8e4m3fn", 0.5) == _scaled("float8e4m3fn", 0.5)
    # A scaled encoding is never equal to the bare format constant
    assert _scaled("float8e4m3fn", 0.5) != _float8e4m3fn
