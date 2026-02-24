# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for converting linear layers to conv"""

from abc import abstractmethod, ABC
import functools
import torch


class AdaptedModule(ABC):
    @abstractmethod
    def adapt(self):
        pass


class ConvInplaceLinear(torch.nn.Conv2d):
    def __init__(self, module: torch.nn.Linear) -> None:
        assert isinstance(module, torch.nn.Linear)
        weight, bias = module.weight, module.bias
        self.out_features, self.in_features = weight.shape

        super().__init__(
            self.in_features,
            self.out_features,
            1,
            dtype=module.weight.dtype,
            bias=bias is not None,
        )

        self.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None and self.bias is not None:
            self.bias.data.copy_(bias.data)
        self.to(module.weight.data.device)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        ndim = x.ndim
        if ndim == 2:
            x = (
                x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1)
            )  # (emb_dim, C) -> (1, C, 1, emb_dim)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(
                0, 2, 3, 1
            )  # (B, emb_dim, C) -> (B, C, 1, emb_dim)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} could not handle input with shape {x.shape}"
            )

        x = super().forward(x)

        if ndim == 2:
            return (
                x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0)
            )  # (1, C, 1, emb_dim) -> # (emb_dim, C)
        if ndim == 3:
            return x.permute(0, 3, 1, 2).squeeze(
                -1
            )  # (1, C, 1, emb_dim) -> # (B, emb_dim, C)
        if ndim == 4:
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return x


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def replace_linears_with_convs(model: torch.nn.Module) -> torch.nn.Module:
    """
    Helper function to replace all linear modules with equivalent conv modules
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            conv_layer = ConvInplaceLinear(module)
            rsetattr(model, name, conv_layer)

    return model
