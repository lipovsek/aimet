# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch

from aimet_torch.v2.quantization.float import FloatEncoding


@pytest.fixture
def scale():
    return torch.tensor(1.0)


class TestFloatEncoding:
    @pytest.mark.parametrize("mantissa_bits, exponent_bits", ((5, 10), (23, 8), (4, 8)))
    def test_create_encoding(self, mantissa_bits, exponent_bits, scale):
        """
        When: Create an encoding from a single scale, mantissa_bits, and exponent_bits
        Then: 1) all encoding parameters are torch.Tensor objects
              2) mantissa_bits is the same as passed value
              3) exponent_bits is the same as passed value
              4) bitwidth is mantissa_bits + exponent_bits + 1
              5) mapping is "float"
        """
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, scale)
        assert isinstance(encoding.scale, torch.Tensor)
        assert isinstance(encoding.maxval, torch.Tensor)
        assert encoding.maxval == encoding._finfo.max
        assert encoding.mantissa_bits == mantissa_bits
        assert encoding.exponent_bits == exponent_bits
        assert encoding.bitwidth == mantissa_bits + exponent_bits + 1
        assert encoding.mapping == "float"

    @pytest.mark.cuda()
    @pytest.mark.parametrize(
        "device, new_device", (("cuda:0", "cpu"), ("cpu", "cuda:0"))
    )
    def test_create_encoding_correct_device(self, device, new_device):
        """
        When: Create an encoding with tensors on device
        Then: encoding.scale is on device
        """
        mantissa_bits = 5
        exponent_bits = 10
        scale = torch.tensor(1.0).to(device)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, scale)
        assert encoding.scale.device == torch.device(device)

        """
        When: call encoding.to(new_device)
        Then: 1) original encoding.scale is on device
              2) returned encoding.scale is on new_device
        """
        new_encoding = encoding.to(new_device)
        assert encoding.scale.device == torch.device(device)
        assert new_encoding.scale.device == torch.device(new_device)

    @pytest.mark.parametrize(
        "dtype, new_dtype",
        ((torch.float16, torch.float32), (torch.float32, torch.float16)),
    )
    def test_create_encoding_correct_dtype(self, dtype, new_dtype):
        """
        When: Create an encoding with tensors of type dtype in {torch.float16, torch.float32}
        Then: encoding.scale is dtype
        """
        mantissa_bits = 5
        exponent_bits = 10
        scale = torch.tensor(1.0).to(dtype)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, scale)
        assert encoding.scale.dtype == dtype

        """
        When: call encoding.to(new_dtype)
        Then: 1) original encoding.scale is dtype
              2) returned encoding.scale is new_dtype
        """
        new_encoding = encoding.to(new_dtype)
        assert encoding.scale.dtype == dtype
        assert new_encoding.scale.dtype == new_dtype

    @pytest.mark.parametrize("shape", ((10, 1), (10,), (1,)))
    def test_perchannel_encoding(self, shape):
        """
        When: Create an encoding with scale whose shape has more than one element
        Then: encoding.scale have shape == shape
              and granularity == "perchannel"
        """
        mantissa_bits = 5
        exponent_bits = 10
        scale = torch.randn(shape)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, scale)
        assert encoding.scale.shape == shape
        assert encoding.granularity == "perchannel"
        assert encoding.mapping == "float"

    def test_pertensor_encoding(self):
        """
        When: Create an encoding with 0-D scale
        Then: encoding.scale have shape == shape
              and granularity == "pertensor"
        """
        mantissa_bits = 5
        exponent_bits = 10
        scale = torch.randn([])
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, scale)
        assert encoding.scale.shape == tuple()
        assert encoding.granularity == "pertensor"
        assert encoding.mapping == "float"
