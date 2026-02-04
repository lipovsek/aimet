# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=redefined-builtin
"""Common utility functions"""

from typing import Callable, Tuple, Any
import functools
import itertools
from packaging import version

import torch


def _is_expandable(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be expanded as target shape
    """
    if len(src_shape) > len(target_shape):
        return False

    for src_dim, dst_dim in zip(src_shape[::-1], target_shape[::-1]):
        if src_dim not in (1, dst_dim):
            return False

    return True


def _is_reducible(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be reduced as target shape
    """
    return _is_expandable(target_shape, src_shape)  # pylint: disable=arguments-out-of-order


def reduce(
    input: torch.Tensor,
    shape: tuple[int, ...],
    reduce_op: Callable,
    block_size: tuple[int, ...] | None = None,
):
    """
    Reduce input into given shape.

    :param input: Input to reduce
    :param shape: Shape of the reduced output
    :param reduce_op: Reduce operation
    :param block_size: Block size for block-wise reduction
    """
    from .quantization._utils import interleave, concretize_block_size

    output_shape = shape

    if block_size is not None:
        block_size = concretize_block_size(input.shape, shape, block_size)
        input = input.reshape(-1, *interleave(shape, block_size))
        shape = interleave(shape, 1)

    if not _is_reducible(input.shape, shape):
        raise RuntimeError(
            f"Input of shape {list(input.shape)} can't be reduced to shape {list(shape)}"
        )

    padded_shape = (*itertools.repeat(1, len(input.shape) - len(shape)), *shape)
    reduce_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
    other_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim > 1)
    permute_dims = reduce_dims + other_dims

    return reduce_op(
        input.permute(permute_dims).reshape(-1, *output_shape), dim=0, keepdim=False
    )


class _ContextManager:
    def __init__(self, action: Callable[[], Any], cleanup: Callable[[], Any]):
        self._action = action
        self._cleanup = cleanup

    def __enter__(self):
        self._action()
        return self

    def __exit__(self, *_):
        self._cleanup()

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return wrapper


class _NullAttribute:
    pass


def patch_attr(obj, attr_name, new_attr) -> _ContextManager:
    """
    Temporarily overwrite object attribute
    """
    if isinstance(obj, torch.nn.Module):
        if attr_name in obj._parameters or attr_name in obj._buffers:  # pylint: disable=protected-access
            return _patch_param_or_buffer(obj, attr_name, new_attr)

    if hasattr(obj, attr_name):
        old_attr = getattr(obj, attr_name)
    else:
        old_attr = _NullAttribute()
    action = lambda: setattr(obj, attr_name, new_attr)

    def cleanup():
        try:
            delattr(obj, attr_name)
        except AttributeError:
            pass

        if not hasattr(obj, attr_name) and not isinstance(old_attr, _NullAttribute):
            setattr(obj, attr_name, old_attr)

    return _ContextManager(action, cleanup)


def _patch_param_or_buffer(
    module: torch.nn.Module,
    param_or_buffer_name: str,
    new_param_or_buffer: torch.Tensor,
):
    """
    Temporarily substitute the reference to the a parameter with the quantized parameter.
    Under the scope of this function, ``getattr(module, param_or_buffer_name)`` will return
    ``new_param_or_buffer`` instead of the original parameter.

    :param module: Module that owns the parameter
    :param param_or_buffer_name: Name of the parameter
    :param new_param_or_buffer: New parameter to replace the original parameter
    """
    # pylint: disable=protected-access

    orig_param_or_buffer = getattr(module, param_or_buffer_name)
    if orig_param_or_buffer is not None:
        assert new_param_or_buffer.shape == orig_param_or_buffer.shape

    if param_or_buffer_name in module._parameters:
        container = module._parameters
    elif param_or_buffer_name in module._buffers:
        container = module._buffers
    elif param_or_buffer_name in module.__dict__:
        # Some non-standard modules (e.g. replicas of torch.nn.DataParallel) store their parameters
        container = module.__dict__
    else:
        raise RuntimeError(
            f"'{param_or_buffer_name}' is not a valid name of parameter of buffer of {type(module)}."
        )

    action = lambda: container.update({param_or_buffer_name: new_param_or_buffer})
    cleanup = lambda: container.update({param_or_buffer_name: orig_param_or_buffer})

    return _ContextManager(action, cleanup)


class _StraightThroughEstimator(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, op, *args, **kwargs):  # pylint:disable=arguments-differ, unused-argument
        return op(*args, **kwargs)

    @staticmethod
    def backward(ctx, *grad):
        return (None, *grad)


def ste_round(*args, **kwargs):
    """
    Applies straight-through rounding
    """
    return _StraightThroughEstimator.apply(torch.round, *args, **kwargs)


class StatisticsNotFoundError(RuntimeError):
    """
    Error raised when compute_encodings() is invoked without statistics
    """


_ENABLE_RECOMPUTE = False


def _set_enable_recompute(mode: bool):
    original_mode = _ENABLE_RECOMPUTE

    def action():
        global _ENABLE_RECOMPUTE  # pylint: disable=global-statement
        _ENABLE_RECOMPUTE = mode

    def cleanup():
        global _ENABLE_RECOMPUTE  # pylint: disable=global-statement
        _ENABLE_RECOMPUTE = original_mode

    return _ContextManager(action, cleanup)


def is_recompute_enabled():
    """
    Returns True if recomputation for memory saving is enabled; False otherwise.
    """
    return _ENABLE_RECOMPUTE


def enable_recompute():
    """
    Enable recomputation for memory saving.
    """
    return _set_enable_recompute(True)


def no_recompute():
    """
    Disable recomputation for memory saving.
    """
    return _set_enable_recompute(False)


def allow_recompute(fn):
    """
    Allow recomputation of activation of the given function during training
    if recompute is enabled.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_recompute_enabled():
            # Enable activation recompute (a.k.a. activataion checkpointing)
            # to reduce memory footprint of training
            return torch.utils.checkpoint.checkpoint(
                fn, *args, use_reentrant=False, **kwargs
            )
        return fn(*args, **kwargs)

    return wrapper


def flatten_nn_module_list(module):
    """
    Flatten nested list of nn.Modules into a flat list
    """

    def flat_iter(mod):
        if isinstance(mod, (list, tuple, torch.nn.ModuleList)):
            for x in mod:
                yield from flat_iter(x)
        else:
            yield mod

    return list(flat_iter(module))


def _map_qmodule(modules, func):
    # pylint: disable=import-outside-toplevel
    # pylint: disable=protected-access, cyclic-import
    from aimet_torch.v2.nn import BaseQuantizationMixin

    contexts = []
    ctx = _ContextManager(
        action=lambda: None,
        cleanup=lambda: [context._cleanup() for context in contexts],
    )

    if isinstance(modules, torch.nn.Module):
        modules = [modules]

    try:
        for module_elem in modules:
            for module in module_elem.modules():
                if isinstance(module, BaseQuantizationMixin):
                    context = func(module)
                    contexts.append(context)
    except Exception:
        ctx._cleanup()
        raise

    return ctx


def remove_input_quantizers(modules):
    """
    Temporarily remove all input quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_input_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_input_quantizers())


def remove_output_quantizers(modules):
    """
    Temporarily remove all output quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_output_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_output_quantizers())


def remove_param_quantizers(modules):
    """
    Temporarily remove all parameter quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_param_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_param_quantizers())


def remove_activation_quantizers(modules):
    """
    Temporarily remove all input and output quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_activation_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    if not isinstance(modules, torch.nn.Module):
        # Shallow copy in case modules is an iterator
        modules = list(modules)

    context_1 = remove_input_quantizers(modules)
    context_2 = remove_output_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(
        action=lambda: None,
        cleanup=lambda: (context_1._cleanup(), context_2._cleanup()),
    )


def remove_all_quantizers(modules):
    """
    Temporarily remove all quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_all_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    if not isinstance(modules, torch.nn.Module):
        # Shallow copy in case modules is an iterator
        modules = list(modules)

    context_1 = remove_activation_quantizers(modules)
    context_2 = remove_param_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(
        action=lambda: None,
        cleanup=lambda: (context_1._cleanup(), context_2._cleanup()),
    )


def has_no_quantizers(module, ignore_params: bool = False) -> bool:
    """
    Helper function to check if a module has any quantizers enabled
    """
    return (
        all(inp_qtzr is None for inp_qtzr in module.input_quantizers)
        and all(out_qtzr is None for out_qtzr in module.output_quantizers)
        and (
            ignore_params
            or all(
                param_qtzr is None for param_qtzr in module.param_quantizers.values()
            )
        )
    )


def rgetattr(obj, attr):
    """Drop in replacement for __getattr__ that can handle dotted attribute strings"""
    return functools.reduce(getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    """Drop in replacement for __setattr__ that can handle dotted attribute strings"""
    pre, _, post = attr.rpartition(".")
    pre_obj = rgetattr(obj, pre) if pre else obj
    return setattr(pre_obj, post, val)


def apply_fn_recursively_to_all_elems(fn, container):
    """Apply fn to all elements in recursively composed container"""
    if container is None:
        return None
    if isinstance(container, (list, tuple)):
        return [apply_fn_recursively_to_all_elems(fn, elem) for elem in container]
    if isinstance(container, dict):
        return {
            key: apply_fn_recursively_to_all_elems(fn, elem)
            for key, elem in container.items()
        }
    return fn(container)


def flatten_list(container):
    """Helper function to flatten nested list/tuple into 1D"""
    if not container:
        return container
    if not isinstance(container, (list, tuple)):
        return [container]
    if isinstance(container[0], (list, tuple)):
        return flatten_list(container[0]) + flatten_list(container[1:])
    if len(container) == 1:
        return container
    return container[:1] + flatten_list(container[1:])


def default_forward_fn(model, inputs):
    """
    Default forward function.
    :param model: pytorch model
    :param inputs: model inputs
    """
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    return model(*inputs)


_torch_compiler_is_compiling: Callable[[], bool]
_torch_compiler_is_dynamo_compiling: Callable[[], bool]
_torch_compiler_is_exporting: Callable[[], bool]

if version.parse(torch.__version__) >= version.parse("2.7"):
    _torch_compiler_is_compiling = torch.compiler.is_compiling
    _torch_compiler_is_dynamo_compiling = torch.compiler.is_dynamo_compiling
    _torch_compiler_is_exporting = torch.compiler.is_exporting
else:
    # torch < 2.7.0 doesn't have torch.compiler.is_compiling/exporting API
    def _torch_compiler_is_compiling() -> bool:
        return False

    def _torch_compiler_is_dynamo_compiling() -> bool:
        return False

    def _torch_compiler_is_exporting() -> bool:
        return False
