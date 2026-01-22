# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""
Tensor factory utility method
"""

import functools
import torch


@functools.lru_cache
def constant_tensor_factory(
    data, *, dtype=None, device=None, pin_memory=False
) -> torch.Tensor:
    """
    Factory function that returns a cached constant scalar tensor with given data.
    This function is intentionally aligned with ``torch.tensor`` API except that
    it doesn't take ``requires_grad`` parameter.

    NOTE: The returned tensor is strictly read-only. However, currently there is no way
          one can fundamentally prevent performing in-place oeprations to the returned
          tensors. It is up to the callers to make sure the returned tensors are not
          modified in-place in any cases.

    :param data: Data for the return tensor.
    :param dtype: Data type of the returned tensor.
    :param device: Device of the returned tensor.
    :param pin_memory: Whether to allocate the tensor to pinned memory. Valid only for CPU tensors.
    :return: Constant read-only tensor
    """
    if not isinstance(data, (float, int)):
        raise ValueError(
            f"Expected data to be an instance of float or int. Got {type(data)})."
        )

    return torch.tensor(
        data, dtype=dtype, device=device, requires_grad=False, pin_memory=pin_memory
    )


def constant_like(data, tensor):
    """
    Factory function that returns a cached constant scalar tensor with the same
    dtype and device as the given tensor. If the input tensor is allocated in
    the pinned memory, the returned tensor will be also allocated to the pinned memory.

    :param data: Data for the return tensor.
    :param tensor: Tensor which the returned tensors will copy properties from.
    :return: Constant read-only tensor
    """
    return constant_tensor_factory(
        data, dtype=tensor.dtype, device=tensor.device, pin_memory=tensor.is_pinned()
    )
