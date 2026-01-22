# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utils for handling custom tensor types"""

try:
    import spconv.pytorch as spconv
except ImportError as e:

    def to_torch_tensor(tensors):
        """placeholder in case spconv doesn't exist"""
        return tensors

    def to_custom_tensor(original, torch_tensors):
        """placeholder in case spconv doesn't exist"""
        return torch_tensors
else:
    from typing import List, Union, Tuple
    import torch

    def to_torch_tensor(original: Union[List, Tuple]) -> List[torch.Tensor]:
        """
        Convert custom tensors to torch tensors
        :param original: List of original tensors
        :return: List of tensors in torch tensor type
        """

        outputs = []

        for tensor in original:
            if isinstance(tensor, spconv.SparseConvTensor):
                tensor = tensor.features
            outputs.append(tensor)

        return outputs

    def to_custom_tensor(
        original: Union[List, Tuple], torch_tensors: List[torch.Tensor]
    ) -> List:
        """
        Convert torch tensors to original custom tensors
        :param original: List of original tensors
        :param torch_tensors: List of torch tensors
        :return: List of tensors in original type
        """

        outputs = []

        for orig, torch_tensor in zip(original, torch_tensors):
            tensor = torch_tensor
            if isinstance(orig, spconv.SparseConvTensor):
                tensor = orig.replace_feature(torch_tensor)
            outputs.append(tensor)

        return outputs
