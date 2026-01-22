# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities for winnowing"""

from typing import List, Tuple
import torch
from torch import nn

from aimet_torch.common.polyslice import PolySlice


class DownsampleLayer(nn.Module):
    """
    This layer down samples the input in the channel dimension according to a given mask.
    Taken from the Qrunchy repo.
    """

    def __init__(self, keep_indices_tensor):
        """keep_indices_tensor: list of indices that should be kept."""
        super().__init__()
        self.keep_tensor = keep_indices_tensor

    # pylint: disable=arguments-differ
    def forward(self, x):
        """Only the needed indices are kept."""
        self.keep_tensor = self.keep_tensor.to("cuda" if x.is_cuda else "cpu")
        y = torch.index_select(x, 1, self.keep_tensor)
        return y

    def get_mask_hard(self):
        """Returns the mask for channels with zero planes.
        A 1 in the mask represents  a channel with zero planes."""
        return self.mask


class UpsampleLayer(nn.Module):
    """
    This layer up samples the input in the channel dimension according to a given mask. If a scale
    is given it also scales the up sampled output.
    """

    def __init__(self, mask, scale=None):
        super().__init__()
        self.register_buffer("mask", mask)
        self.register_buffer("indices", mask.nonzero().squeeze(1))
        if scale is not None:
            self.register_buffer("scale", scale[None, :, None, None])

    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward  function."""
        out = torch.zeros(
            x.shape[0], self.mask.shape[0], x.shape[2], x.shape[3], device=x.device
        )
        out = out.to("cuda" if x.is_cuda else "cpu")
        if x.dtype == torch.float64:
            out = out.double()
        out[:, self.indices] = x
        return out * self.scale if hasattr(self, "scale") else out


class ReShape(nn.Module):
    """ReShape layer class that reshapes the input to the desired shape."""

    def __init__(self, *args):
        """Desired shape is passed in as argument."""
        super().__init__()
        self.shape = args

    # pylint: disable=arguments-differ
    def forward(self, x):
        """forward pass"""
        return x.view(self.shape)


def zero_out_input_channels(module, input_channels_to_prune):
    """
    :param module: module
    :param input_channels_to_prune: list of input channels to be zeroed out
    :return:
    """
    for input_channel in input_channels_to_prune:
        if isinstance(module, nn.Conv2d):
            module.weight.data[:, input_channel, :, :] = 0

        elif isinstance(module, nn.Linear):
            module.weight.data[:, input_channel] = 0

        else:
            raise ValueError("Unsupported layer_type")


def search_for_zero_planes(
    model: torch.nn.Module,
) -> List[Tuple[torch.nn.Module, List[int]]]:
    """If list of modules to winnow is empty to start with, search through all modules to check if any
    planes have been zeroed out. Update self._list_of_modules_to_winnow with any findings.
    :param model: torch model to search through modules for zeroed parameters
    """

    list_of_modules_to_winnow = []
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv.Conv2d)):
            in_channels_to_winnow = _assess_weight_and_bias(module.weight, module.bias)
            if in_channels_to_winnow:
                list_of_modules_to_winnow.append((module, in_channels_to_winnow))
    return list_of_modules_to_winnow


def _assess_weight_and_bias(weight: torch.nn.Parameter, _bias: torch.nn.Parameter):
    """4-dim weights [CH-out, CH-in, H, W] and 1-dim bias [CH-out]"""
    num_out = weight.shape[0]
    num_in = weight.shape[1]

    input_channels_to_ignore = []
    for ch_in in range(num_in):
        can_winnow = True
        for ch_out in range(num_out):
            if not _is_zero_tensor(weight[ch_out, ch_in]):
                can_winnow = False
                break
        if can_winnow:
            input_channels_to_ignore.append(ch_in)

    # Enable when implementing output pruning:
    # return input_channels_to_ignore, outputs_to_ignore

    return input_channels_to_ignore


def _is_zero_tensor(tensor: torch.Tensor):
    return (tensor != 0.0).any().item() == 0


def reduce_tensor(tensor: torch.Tensor, reduction: PolySlice):
    """Removes slices in one or more of the tensor's dimensions."""
    using_cuda = bool(tensor.is_cuda)

    result = torch.tensor(tensor)  # pylint: disable=not-callable

    for dim, index in reduction.get_all().items():
        assert dim <= len(tensor.shape)
        dim_size = tensor.shape[dim]
        assert max(index) < dim_size
        to_keep = [i for i in range(dim_size) if i not in index]
        if using_cuda:
            result = torch.index_select(result, dim, torch.tensor(to_keep).cuda())  # pylint: disable=not-callable
        else:
            result = torch.index_select(result, dim, torch.tensor(to_keep))  # pylint: disable=not-callable
    return result


def get_one_positions_in_binary_mask(mask):
    """
    Return the indices of one positions in a binary mask.

    :param mask: a mask that contains either 0s or 1s
    :return:
    """
    mask_one_positions = [i for i in range(len(mask)) if mask[i] == 1]
    return mask_one_positions
