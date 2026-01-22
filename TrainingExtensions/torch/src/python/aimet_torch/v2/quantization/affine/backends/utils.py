# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=all
import torch
from aimet_torch.v2.quantization.affine.backends import torch_builtins

from typing import List, Optional, Protocol
from aimet_torch.v2.utils import _ContextManager


class _QuantizationBackendProtocol(Protocol):
    def quantize(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        block_size: Optional[List] = None,
    ) -> torch.Tensor: ...

    def dequantize(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        block_size: Optional[List] = None,
    ) -> torch.Tensor: ...

    def quantize_dequantize(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        block_size: Optional[List] = None,
        zero_point_shift: float = 0.0,
    ) -> torch.Tensor: ...


_CURRENT_BACKEND = "torch_builtins"

_SUPPORTED_BACKENDS = {
    "torch_builtins": torch_builtins,
}


def set_global_backend(name: str):
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = name


def set_backend(name: str) -> _ContextManager:
    if name not in _SUPPORTED_BACKENDS:
        supported_backend_names = ", ".join(_SUPPORTED_BACKENDS.keys())
        raise RuntimeError(
            f"Backend '{name}' is not supported. "
            f"Please choose one of: {supported_backend_names}"
        )

    old_backend = _CURRENT_BACKEND
    action = lambda: set_global_backend(name)
    cleanup = lambda: set_global_backend(old_backend)
    return _ContextManager(action=action, cleanup=cleanup)


def get_backend() -> _QuantizationBackendProtocol:
    return _SUPPORTED_BACKENDS[_CURRENT_BACKEND]


def add_backend(name: str, module: _QuantizationBackendProtocol):
    if name in _SUPPORTED_BACKENDS:
        return RuntimeError(f"{name} is exist.")

    _SUPPORTED_BACKENDS[name] = module


__all__ = ["set_global_backend", "set_backend", "get_backend", "add_backend"]
