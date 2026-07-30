# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from aimet_onnx.defs import QSpec, PerChannel, PerTensor, Blockwise, LPBQ
from aimet_onnx import int2, int4, int8, float16


def test_qspec_per_tensor():
    spec = QSpec.per_tensor("int4")
    assert isinstance(spec.granularity, PerTensor)
    assert spec.dtype == int4
    assert spec.symmetric is None
    assert not spec.shift_zero_point

    spec = QSpec.per_tensor(int8, symmetric=False)
    assert isinstance(spec.granularity, PerTensor)
    assert spec.dtype == int8
    assert spec.symmetric is False
    assert not spec.shift_zero_point


def test_qspec_per_channel():
    spec = QSpec.per_channel("int8", symmetric=True)
    assert isinstance(spec.granularity, PerChannel)
    assert spec.dtype == int8
    assert spec.symmetric
    assert not spec.shift_zero_point


def test_qspec_shifted_zero_point():
    spec = QSpec.per_channel("int2", symmetric=True, shift_zero_point=True)
    assert spec.dtype == int2
    assert spec.symmetric
    assert spec.shift_zero_point

    with pytest.raises(ValueError):
        spec = QSpec.per_channel("int4", symmetric=True, shift_zero_point=True)

    with pytest.raises(ValueError):
        spec = QSpec.per_channel("int2", symmetric=False, shift_zero_point=True)


def test_qspec_blockwise():
    spec = QSpec.blockwise(int4, 64, symmetric=True)
    assert spec.dtype == int4
    assert spec.granularity == Blockwise(64)
    assert not spec.shift_zero_point
    assert spec.symmetric


def test_qspec_lpbq():
    spec = QSpec.lpbq("int4")
    assert isinstance(spec.granularity, LPBQ)
    assert spec.dtype == int4
    assert spec.granularity.block_size == 64
    assert spec.granularity.scale_bits == 4
    assert spec.symmetric

    spec = QSpec.lpbq("int2", block_size=32, scale_bits=2)
    assert isinstance(spec.granularity, LPBQ)
    assert spec.dtype == int2
    assert spec.granularity.block_size == 32
    assert spec.granularity.scale_bits == 2
    assert spec.symmetric

    with pytest.raises(ValueError):
        spec = QSpec.lpbq(float16)

    with pytest.raises(ValueError):
        spec = QSpec(int4, LPBQ(64, 4), symmetric=False)

    with pytest.raises(ValueError):
        spec = QSpec(int4, LPBQ(64, 4), shift_zero_point=True)


def test_qspec_dtype_from_string():
    assert QSpec.per_tensor("int8").dtype == int8

    with pytest.raises(ValueError):
        QSpec.per_tensor("not_a_dtype")


def test_qspec_dtype_invalid_type():
    with pytest.raises(TypeError):
        QSpec.per_tensor(8)
