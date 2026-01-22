# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch

from aimet_torch.v2.quantization.float import FloatEncoding


@pytest.fixture
def maxval():
    return torch.tensor(1.0)


class TestFloatEncoding:
    @pytest.mark.parametrize("mantissa_bits, exponent_bits", ((5, 10), (8, 23), (4, 8)))
    def test_create_encoding(self, mantissa_bits, exponent_bits, maxval):
        """
        When: Create an encoding from a single maxval, mantissa_bits, and exponent_bits
        Then: 1) all encoding parameters are torch.Tensor objects
              2) mantissa_bits is the same as passed value
              3) exponent_bits is the same as passed value
              4) bitwidth is mantissa_bits + exponent_bits + 1
              5) mapping is "float"
        """
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, maxval)
        assert isinstance(encoding.maxval, torch.Tensor)
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
        Then: encoding.maxval is on device
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.tensor(1.0).to(device)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, maxval)
        assert encoding.maxval.device == torch.device(device)

        """
        When: call encoding.to(new_device)
        Then: 1) original encoding.maxval is on device
              2) returned encoding.maxval is on new_device
        """
        new_encoding = encoding.to(new_device)
        assert encoding.maxval.device == torch.device(device)
        assert new_encoding.maxval.device == torch.device(new_device)

    @pytest.mark.parametrize(
        "dtype, new_dtype",
        ((torch.float16, torch.float32), (torch.float32, torch.float16)),
    )
    def test_create_encoding_correct_dtype(self, dtype, new_dtype):
        """
        When: Create an encoding with tensors of type dtype in {torch.float16, torch.float32}
        Then: encoding.maxval is dtype
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.tensor(1.0).to(dtype)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, maxval)
        assert encoding.maxval.dtype == dtype

        """
        When: call encoding.to(new_dtype)
        Then: 1) original encoding.maxval is dtype
              2) returned encoding.maxval is new_dtype
        """
        new_encoding = encoding.to(new_dtype)
        assert encoding.maxval.dtype == dtype
        assert new_encoding.maxval.dtype == new_dtype

    @pytest.mark.parametrize("shape", ((10, 1), (10,), (1,)))
    def test_perchannel_encoding(self, shape):
        """
        When: Create an encoding with maxval whose shape has more than one element
        Then: encoding.maxval have shape == shape
              and granularity == "perchannel"
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.randn(shape)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, maxval)
        assert encoding.maxval.shape == shape
        assert encoding.granularity == "perchannel"
        assert encoding.mapping == "float"

    def test_pertensor_encoding(self):
        """
        When: Create an encoding with 0-D maxval
        Then: encoding.maxval have shape == shape
              and granularity == "pertensor"
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.randn([])
        encoding = FloatEncoding(mantissa_bits, exponent_bits, False, False, maxval)
        assert encoding.maxval.shape == tuple()
        assert encoding.granularity == "pertensor"
        assert encoding.mapping == "float"
