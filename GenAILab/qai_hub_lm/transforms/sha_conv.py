# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
SHA with Conv replacement adaptations.

Combines Split-Head Attention (SHA) with Linear-to-Conv2d replacement
for models that benefit from both transformations.
"""

import torch

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.transforms.base import rsetattr
from GenAILab.qai_hub_lm.transforms.sha import LlamaSHAMixin, Qwen3SHAMixin


# ============================================================================
# Conv replacement utilities
# ============================================================================


class ConvInplaceLinear(torch.nn.Conv2d):
    """Conv2d layer that replaces a Linear layer with equivalent functionality."""

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
            x = x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(0, 2, 3, 1)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} could not handle input with shape {x.shape}"
            )

        x = super().forward(x)

        if ndim == 2:
            return x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0)
        if ndim == 3:
            return x.permute(0, 3, 1, 2).squeeze(-1)
        if ndim == 4:
            x = x.permute(0, 2, 3, 1)
        return x


def replace_linears_with_convs(model: torch.nn.Module) -> torch.nn.Module:
    """Replace all Linear modules with equivalent Conv2d modules."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            conv_layer = ConvInplaceLinear(module)
            rsetattr(model, name, conv_layer)
    return model


# ============================================================================
# SHA + Conv Mixins
# ============================================================================


class LlamaSHAConvMixin(LlamaSHAMixin):
    """Mixin to enable SHA + Conv replacement for Llama models."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs):
        return replace_linears_with_convs(super().instantiate_model(*args, **kwargs))


class Qwen3SHAConvMixin(Qwen3SHAMixin):
    """Mixin to enable SHA + Conv replacement for Qwen3 models."""

    @classmethod
    def instantiate_model(cls, *args, **kwargs):
        return replace_linears_with_convs(super().instantiate_model(*args, **kwargs))


# ============================================================================
# Registered SHA_Conv Adaptations
# ============================================================================


@YAMLConfigParser.register_adaptation("SHA_Conv", model_type="llama", exclusive=True)
class LlamaSHAConvAdaptation(LlamaSHAConvMixin):
    """SHA with Conv adaptation for Llama models."""

    pass


@YAMLConfigParser.register_adaptation("SHA_Conv", model_type="qwen3", exclusive=True)
class Qwen3SHAConvAdaptation(Qwen3SHAConvMixin):
    """SHA with Conv adaptation for Qwen3 models."""

    pass
