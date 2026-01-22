# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import unittest
import torch
from aimet_common.polyslice import PolySlice
from aimet_torch.winnow.winnow_utils import reduce_tensor


def tensor_contains(tensor, value):
    return (tensor == value).nonzero().numel() > 0


class TestTrainingExtensionsTensorReduction(unittest.TestCase):
    def test_tensor_reduction(self):
        shape = [3, 2, 4]
        tensor = torch.zeros(shape, dtype=torch.int8)
        view = tensor.reshape([-1])
        for i in range(tensor.numel()):
            view[i] = 101 + i

        reduct = PolySlice(dim=0, index=1)
        result = reduce_tensor(tensor, reduct)
        assert list(result.shape) == [2, 2, 4]

        assert tensor_contains(result, 101)
        assert tensor_contains(result, 108)
        assert not tensor_contains(result, 109)
        assert not tensor_contains(result, 116)
        assert tensor_contains(result, 117)
        assert tensor_contains(result, 124)

        reduct.set(dim=2, index=[0])
        reduct.add(dim=2, index=3)
        result = reduce_tensor(tensor, reduct)
        assert list(result.shape) == [2, 2, 2]

        assert not tensor_contains(result, 101)
        assert tensor_contains(result, 102)
        assert tensor_contains(result, 103)
        assert not tensor_contains(result, 104)

        assert not tensor_contains(result, 117)
        assert not tensor_contains(result, 121)
        assert tensor_contains(result, 122)
        assert tensor_contains(result, 123)
        assert not tensor_contains(result, 124)
