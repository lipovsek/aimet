# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=all
import torch
from aimet_torch.v2.quantization.affine.backends import (
    torch_builtins,
    triton as _triton,
)

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

if _triton.is_available():
    _SUPPORTED_BACKENDS["triton"] = _triton


def set_backend(name: str) -> _ContextManager:
    """
    Set global backend for quantization operations.
    Choices: ["triton", "torch_builtins"]

    Example:
        >>> # Temporarily set backend to triton
        >>> with aimet_torch.quantization.set_backend("triton"):
        ...     aimet_torch.quantization.affine.quantize(
        ...         torch.arange(0, 1, step=0.1), torch.tensor(0.005), torch.tensor(0), 0, 255,
        ...     )
        ...
        tensor([  0.,  20.,  40.,  60.,  80., 100., 120., 140., 160., 180.])
        >>> # Permanently set backend to triton
        >>> aimet_torch.quantization.set_backend("triton")
        >>> aimet_torch.quantization.affine.quantize(
        ...     torch.arange(0, 1, step=0.1), torch.tensor(0.005), torch.tensor(0), 0, 255,
        ... )
        ...
        tensor([  0.,  20.,  40.,  60.,  80., 100., 120., 140., 160., 180.])
    """
    global _CURRENT_BACKEND

    if name not in _SUPPORTED_BACKENDS:
        supported_backend_names = ", ".join(_SUPPORTED_BACKENDS.keys())
        raise RuntimeError(
            f"Backend '{name}' is not supported. "
            f"Please choose one of: {supported_backend_names}"
        )

    old_backend = _CURRENT_BACKEND
    _CURRENT_BACKEND = name

    def cleanup():
        global _CURRENT_BACKEND
        _CURRENT_BACKEND = old_backend

    return _ContextManager(action=lambda: None, cleanup=cleanup)


def get_backend() -> _QuantizationBackendProtocol:
    """
    Get global backend for quantization operations.

    Example:
        >>> aimet_torch.quantization.set_backend("triton")
        >>> aimet_torch.quantization.get_backend().__name__
        'aimet_torch.v2.quantization.affine.backends.triton'
    """
    return _SUPPORTED_BACKENDS[_CURRENT_BACKEND]


def add_backend(name: str, module: _QuantizationBackendProtocol):
    if name in _SUPPORTED_BACKENDS:
        return RuntimeError(f"Backend {name} already exists.")

    _SUPPORTED_BACKENDS[name] = module


__all__ = ["set_backend", "get_backend", "add_backend"]
