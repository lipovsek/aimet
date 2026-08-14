# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Callable
import torch
from torch._decomp import core_aten_decompositions


@functools.lru_cache
def _qnn_friendly_aten_ops() -> tuple[torch._ops.OpOverloadPacket, ...]:
    """
    List of ATen ops that can be mapped (almost) 1-to-1 to QNN IR
    """
    # pylint: disable=protected-access
    # Commented out are ops that are either 1) unclear whether always 1-to-1 mappable
    # or 2) always 1-to-1 mappable in theory but unclear in practice
    return (
        torch.ops.aten._adaptive_avg_pool2d,
        torch.ops.aten._log_softmax,
        # torch.ops.aten._native_batch_norm_legit,
        # torch.ops.aten._native_batch_norm_legit_no_training,
        torch.ops.aten._prelu_kernel,
        # torch.ops.aten._safe_softmax,
        torch.ops.aten._softmax,
        torch.ops.aten.abs,
        torch.ops.aten.absolute,
        torch.ops.aten.adaptive_avg_pool2d,
        torch.ops.aten.add,
        torch.ops.aten.addmm,
        torch.ops.aten.amax,
        torch.ops.aten.amin,
        torch.ops.aten.avg_pool1d,
        torch.ops.aten.avg_pool2d,
        torch.ops.aten.avg_pool3d,
        torch.ops.aten.bmm,
        torch.ops.aten.cat,
        torch.ops.aten.channel_shuffle,
        torch.ops.aten.clamp,
        torch.ops.aten.col2im,
        torch.ops.aten.constant_pad_nd,
        torch.ops.aten.conv1d,
        torch.ops.aten.conv2d,
        torch.ops.aten.conv3d,
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.conv_transpose3d,
        torch.ops.aten.convolution,
        torch.ops.aten.copy,
        # torch.ops.aten.cudnn_batch_norm,
        torch.ops.aten.detach,
        torch.ops.aten.div,
        torch.ops.aten.divide,
        # torch.ops.aten.dot,
        torch.ops.aten.elu,
        torch.ops.aten.embedding,
        torch.ops.aten.eq,
        torch.ops.aten.exp,
        torch.ops.aten.gather,
        torch.ops.aten.ge,
        torch.ops.aten.gelu,
        torch.ops.aten.greater,
        torch.ops.aten.greater_equal,
        torch.ops.aten.grid_sampler_2d,
        torch.ops.aten.gt,
        torch.ops.aten.hardswish,
        torch.ops.aten.hardtanh,
        torch.ops.aten.im2col,
        torch.ops.aten.index,
        torch.ops.aten.index_add,
        torch.ops.aten.index_copy,
        torch.ops.aten.index_fill,
        torch.ops.aten.index_put,
        torch.ops.aten.index_select,
        torch.ops.aten.instance_norm,
        torch.ops.aten.le,
        torch.ops.aten.leaky_relu,
        torch.ops.aten.less,
        torch.ops.aten.less_equal,
        torch.ops.aten.linear,
        torch.ops.aten.log,
        torch.ops.aten.log_softmax,
        torch.ops.aten.logical_and,
        torch.ops.aten.logical_not,
        torch.ops.aten.logical_or,
        torch.ops.aten.logical_xor,
        torch.ops.aten.lt,
        torch.ops.aten.masked_fill,
        torch.ops.aten.masked_scatter,
        torch.ops.aten.matmul,
        torch.ops.aten.max_pool1d,
        torch.ops.aten.max_pool2d,
        torch.ops.aten.max_pool3d,
        torch.ops.aten.mean,
        torch.ops.aten.mm,
        torch.ops.aten.mul,
        torch.ops.aten.multiply,
        torch.ops.aten.mv,
        torch.ops.aten.narrow,
        # torch.ops.aten.native_batch_norm,
        # torch.ops.aten.native_group_norm,
        torch.ops.aten.native_layer_norm,
        torch.ops.aten.neg,
        torch.ops.aten.negative,
        torch.ops.aten.pixel_shuffle,
        torch.ops.aten.pixel_unshuffle,
        torch.ops.aten.pow,
        torch.ops.aten.prelu,
        torch.ops.aten.reciprocal,
        torch.ops.aten.reflection_pad1d,
        torch.ops.aten.reflection_pad2d,
        torch.ops.aten.reflection_pad3d,
        torch.ops.aten.relu,
        torch.ops.aten.relu6,
        torch.ops.aten.repeat,
        torch.ops.aten.repeat_interleave,
        torch.ops.aten.replication_pad1d,
        torch.ops.aten.replication_pad2d,
        torch.ops.aten.replication_pad3d,
        # torch.ops.aten.rms_norm,
        torch.ops.aten.scatter,
        torch.ops.aten.scatter_add,
        torch.ops.aten.scatter_reduce,
        torch.ops.aten.select,
        torch.ops.aten.sigmoid,
        torch.ops.aten.slice,
        torch.ops.aten.split,
        torch.ops.aten.split_with_sizes,
        torch.ops.aten.split_with_sizes_copy,
        torch.ops.aten.sqrt,
        torch.ops.aten.stack,
        torch.ops.aten.sub,
        torch.ops.aten.subtract,
        torch.ops.aten.sum,
        torch.ops.aten.tanh,
        torch.ops.aten.unbind,
        # torch.ops.aten.upsample_bicubic2d,
        torch.ops.aten.upsample_linear1d,
        # torch.ops.aten.vdot,
        torch.ops.aten.where,
    )


def _qnn_decompositions() -> dict[torch._ops.OperatorBase, Callable]:
    """
    Returns quantization-friendly decomposition table for QNN.
    For the most part, this is a subset of core ATen decomposition table
    to prevent premature over-lowering before quantization

    Conceptually,

                                 Export   (`torch.export`)
                                   |
                          ATen ... |
                                   |
                                   V
                                 Lower    (`from_torch_exported_program`)
                                   |
    quantization-friendly ATen ... |
                                   |
                                   V
                                 Quantize (`compute_missing_encodings`)
                                   |
                quantized ATen ... |
                                   |
                                   V
                                 Lower
                                   |
           quantized core ATen ... |
                                   |
                                   V
                                 (...)
    """
    decomp_table = core_aten_decompositions()
    skiplist = [
        getattr(opoverloadpacket, name)
        for opoverloadpacket in _qnn_friendly_aten_ops()
        for name in opoverloadpacket.overloads()
    ]

    for op in skiplist:
        _ = decomp_table.pop(op, None)

    decomp_table.update(_additional_decomposition_registry)
    return decomp_table


_additional_decomposition_registry: dict[torch._ops.OperatorBase, Callable] = {}


def _register_qnn_decomposition(op: torch._ops.OpOverloadPacket):
    def decorator(decomp: Callable):
        for name in op.overloads():
            _additional_decomposition_registry[getattr(op, name)] = decomp
        return decomp

    return decorator


@_register_qnn_decomposition(torch.ops.aten.relu6)
def relu6(self: torch.Tensor):
    """
    Technically, relu6 doesn't need further decomposition. This decomposition
    is only to promote clamp as the canonical/preferred lowering of relu6
    """
    return torch.ops.aten.clamp(self, min=0.0, max=6.0)
