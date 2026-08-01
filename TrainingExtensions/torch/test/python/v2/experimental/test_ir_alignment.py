# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Callable, Any
import onnx
import torch
import aimet_torch
from aimet_torch.experimental.export.exported_program import (
    ExportedProgram as AimetExportedProgram,
)
from aimet_torch.common.onnx._utils import (
    _is_grid_preserving_op,
    _is_metadata_op,
    _onnx_to_qnn_ir,
)
from aimet_torch.experimental.export.qnn import _qnn_friendly_aten_ops
from torch.utils._pytree import tree_iter
import pytest

sample_inputs_factory: dict[torch._ops.OpOverloadPacket, Callable[[], Any]] = {
    torch.ops.aten._adaptive_avg_pool2d: lambda: (torch.randn(1, 3, 8, 8), [4, 4]),
    torch.ops.aten._log_softmax: lambda: (torch.randn(2, 3), 1, False),
    torch.ops.aten._native_batch_norm_legit: lambda: (
        torch.randn(2, 3, 4, 4),
        torch.ones(3),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3),
        False,
        0.1,
        1e-5,
    ),
    torch.ops.aten._native_batch_norm_legit_no_training: lambda: (
        torch.randn(2, 3, 4, 4),
        torch.ones(3),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3),
        0.1,
        1e-5,
    ),
    torch.ops.aten._prelu_kernel: lambda: (
        torch.randn(2, 3, 4, 4),
        torch.tensor([0.25]),
    ),
    torch.ops.aten._safe_softmax: lambda: (torch.randn(2, 3), 1, None),
    torch.ops.aten._softmax: lambda: (torch.randn(2, 3), 1, False),
    torch.ops.aten.abs: lambda: (torch.randn(2, 3),),
    torch.ops.aten.absolute: lambda: (torch.randn(2, 3),),
    torch.ops.aten.adaptive_avg_pool2d: lambda: (torch.randn(1, 3, 8, 8), [4, 4]),
    torch.ops.aten.add: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.addmm: lambda: (
        torch.randn(2, 3),
        torch.randn(2, 4),
        torch.randn(4, 3),
    ),
    torch.ops.aten.amax: lambda: (torch.randn(2, 3, 4), [1]),
    torch.ops.aten.amin: lambda: (torch.randn(2, 3, 4), [1]),
    torch.ops.aten.avg_pool1d: lambda: (torch.randn(1, 3, 10), [3]),
    torch.ops.aten.avg_pool2d: lambda: (torch.randn(1, 3, 8, 8), [2, 2]),
    torch.ops.aten.avg_pool3d: lambda: (torch.randn(1, 3, 4, 4, 4), [2, 2, 2]),
    torch.ops.aten.bmm: lambda: (torch.randn(5, 3, 4), torch.randn(5, 4, 2)),
    torch.ops.aten.cat: lambda: ([torch.randn(2, 3), torch.randn(2, 3)], 0),
    torch.ops.aten.channel_shuffle: lambda: (torch.randn(1, 4, 8, 8), 2),
    torch.ops.aten.clamp: lambda: (torch.randn(2, 3), -1.0, 1.0),
    torch.ops.aten.col2im: lambda: (
        torch.randn(1, 27, 4),
        [4, 4],
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 1],
    ),
    torch.ops.aten.constant_pad_nd: lambda: (
        torch.randn(1, 3, 4, 4),
        [1, 1, 1, 1],
        0.0,
    ),
    torch.ops.aten.conv1d: lambda: (
        torch.randn(1, 3, 8),
        torch.randn(4, 3, 3),
        None,
        [1],
        [0],
        [1],
        1,
    ),
    torch.ops.aten.conv2d: lambda: (
        torch.randn(1, 3, 8, 8),
        torch.randn(4, 3, 3, 3),
        None,
        [1, 1],
        [0, 0],
        [1, 1],
        1,
    ),
    torch.ops.aten.conv3d: lambda: (
        torch.randn(1, 3, 4, 4, 4),
        torch.randn(4, 3, 3, 3, 3),
        None,
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
    ),
    torch.ops.aten.conv_transpose1d: lambda: (
        torch.randn(1, 3, 8),
        torch.randn(3, 4, 3),
        None,
        [1],
        [0],
        [0],
        1,
        [1],
    ),
    torch.ops.aten.conv_transpose2d: lambda: (
        torch.randn(1, 3, 8, 8),
        torch.randn(3, 4, 3, 3),
        None,
        [1, 1],
        [0, 0],
        [0, 0],
        1,
        [1, 1],
    ),
    torch.ops.aten.conv_transpose3d: lambda: (
        torch.randn(1, 3, 4, 4, 4),
        torch.randn(3, 4, 3, 3, 3),
        None,
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        1,
        [1, 1, 1],
    ),
    torch.ops.aten.convolution: lambda: (
        torch.randn(1, 3, 8, 8),
        torch.randn(4, 3, 3, 3),
        None,
        [1, 1],
        [0, 0],
        [1, 1],
        False,
        [0, 0],
        1,
    ),
    torch.ops.aten.copy: lambda: (torch.randn(2, 3), torch.randn(2, 3), False),
    # torch.ops.aten.cudnn_batch_norm: lambda: (
    #     torch.randn(2, 3, 4, 4),
    #     torch.ones(3),
    #     torch.ones(3),
    #     torch.zeros(3),
    #     torch.zeros(3),
    #     False,
    #     0.1,
    #     1e-5,
    # ),
    torch.ops.aten.detach: lambda: (torch.randn(2, 3),),
    torch.ops.aten.div: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.divide: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.dot: lambda: (torch.randn(5), torch.randn(5)),
    torch.ops.aten.elu: lambda: (torch.randn(2, 3), 1.0, 1.0, 1.0),
    torch.ops.aten.embedding: lambda: (
        torch.randn(10, 3),
        torch.tensor([0, 2, 5]),
        -1,
        False,
        False,
    ),
    torch.ops.aten.eq: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.exp: lambda: (torch.randn(2, 3),),
    torch.ops.aten.gather: lambda: (
        torch.randn(2, 3),
        1,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
    ),
    torch.ops.aten.ge: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.gelu: lambda: (torch.randn(2, 3),),
    torch.ops.aten.greater: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.greater_equal: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.grid_sampler_2d: lambda: (
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 2),
        0,
        0,
        True,
    ),
    torch.ops.aten.gt: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.hardswish: lambda: (torch.randn(2, 3),),
    torch.ops.aten.hardtanh: lambda: (torch.randn(2, 3), -1.0, 1.0),
    torch.ops.aten.im2col: lambda: (
        torch.randn(1, 3, 4, 4),
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 1],
    ),
    torch.ops.aten.index: lambda: (
        torch.randn(3, 4, 5),
        [torch.tensor([0, 2]), None, torch.tensor([1, 3])],
    ),
    torch.ops.aten.index_add: lambda: (
        torch.randn(3, 4),
        0,
        torch.tensor([0, 2]),
        torch.randn(2, 4),
    ),
    torch.ops.aten.index_copy: lambda: (
        torch.randn(3, 4),
        0,
        torch.tensor([0, 2]),
        torch.randn(2, 4),
    ),
    torch.ops.aten.index_fill: lambda: (
        torch.randn(3, 4),
        0,
        torch.tensor([0, 2]),
        1.0,
    ),
    torch.ops.aten.index_put: lambda: (
        torch.randn(3, 4),
        [torch.tensor([0, 2])],
        torch.randn(2, 4),
        False,
    ),
    torch.ops.aten.index_select: lambda: (
        torch.randn(3, 4),
        0,
        torch.tensor([0, 2]),
    ),
    torch.ops.aten.instance_norm: lambda: (
        torch.randn(2, 3, 4, 4),
        torch.ones(3),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3),
        True,
        0.1,
        1e-5,
        False,
    ),
    torch.ops.aten.le: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.leaky_relu: lambda: (torch.randn(2, 3), 0.01),
    torch.ops.aten.less: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.less_equal: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.linear: lambda: (torch.randn(2, 3), torch.randn(4, 3), None),
    torch.ops.aten.log: lambda: (torch.randn(2, 3).abs() + 1,),
    torch.ops.aten.log_softmax: lambda: (torch.randn(2, 3), 1),
    torch.ops.aten.logical_and: lambda: (torch.randn(2, 3) > 0, torch.randn(2, 3) > 0),
    torch.ops.aten.logical_not: lambda: (torch.randn(2, 3) > 0,),
    torch.ops.aten.logical_or: lambda: (torch.randn(2, 3) > 0, torch.randn(2, 3) > 0),
    torch.ops.aten.logical_xor: lambda: (torch.randn(2, 3) > 0, torch.randn(2, 3) > 0),
    torch.ops.aten.lt: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.masked_fill: lambda: (
        torch.randn(2, 3),
        torch.randn(2, 3) > 0,
        0.0,
    ),
    torch.ops.aten.masked_scatter: lambda: (
        torch.randn(2, 3),
        torch.arange(6).reshape(2, 3) < 3,
        torch.randn(3),
    ),
    torch.ops.aten.matmul: lambda: (torch.randn(2, 3), torch.randn(3, 4)),
    torch.ops.aten.max_pool1d: lambda: (torch.randn(1, 3, 10), [3]),
    torch.ops.aten.max_pool2d: lambda: (torch.randn(1, 3, 8, 8), [2, 2]),
    torch.ops.aten.max_pool3d: lambda: (torch.randn(1, 3, 4, 4, 4), [2, 2, 2]),
    torch.ops.aten.mean: lambda: (torch.randn(2, 3, 4),),
    torch.ops.aten.mm: lambda: (torch.randn(2, 3), torch.randn(3, 4)),
    torch.ops.aten.mul: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.multiply: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.mv: lambda: (torch.randn(3, 4), torch.randn(4)),
    torch.ops.aten.narrow: lambda: (torch.randn(3, 4, 5), 1, 1, 2),
    torch.ops.aten.native_batch_norm: lambda: (
        torch.randn(2, 3, 4, 4),
        torch.ones(3),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3),
        False,
        0.1,
        1e-5,
    ),
    torch.ops.aten.native_group_norm: lambda: (
        torch.randn(2, 4, 5),
        torch.ones(4),
        torch.zeros(4),
        2,
        4,
        5,
        2,
        1e-5,
    ),
    torch.ops.aten.native_layer_norm: lambda: (
        torch.randn(2, 3, 4),
        [4],
        torch.ones(4),
        torch.zeros(4),
        1e-5,
    ),
    torch.ops.aten.neg: lambda: (torch.randn(2, 3),),
    torch.ops.aten.negative: lambda: (torch.randn(2, 3),),
    torch.ops.aten.pixel_shuffle: lambda: (torch.randn(1, 4, 8, 8), 2),
    torch.ops.aten.pixel_unshuffle: lambda: (torch.randn(1, 4, 8, 8), 2),
    torch.ops.aten.pow: lambda: (torch.randn(2, 3), 2.0),
    torch.ops.aten.prelu: lambda: (torch.randn(2, 3, 4, 4), torch.ones(3) * 0.25),
    torch.ops.aten.reciprocal: lambda: (torch.randn(2, 3) + 2.0,),
    torch.ops.aten.reflection_pad1d: lambda: (torch.randn(1, 3, 8), [2, 2]),
    torch.ops.aten.reflection_pad2d: lambda: (torch.randn(1, 3, 4, 4), [1, 1, 1, 1]),
    torch.ops.aten.reflection_pad3d: lambda: (
        torch.randn(1, 3, 4, 4, 4),
        [1, 1, 1, 1, 1, 1],
    ),
    torch.ops.aten.relu6: lambda: (torch.randn(2, 3),),
    torch.ops.aten.relu: lambda: (torch.randn(2, 3),),
    torch.ops.aten.repeat: lambda: (torch.randn(2, 3), [2, 1]),
    torch.ops.aten.repeat_interleave: lambda: (torch.randn(2, 3), 2, 0),
    torch.ops.aten.replication_pad1d: lambda: (torch.randn(1, 3, 8), [2, 2]),
    torch.ops.aten.replication_pad2d: lambda: (torch.randn(1, 3, 4, 4), [1, 1, 1, 1]),
    torch.ops.aten.replication_pad3d: lambda: (
        torch.randn(1, 3, 4, 4, 4),
        [1, 1, 1, 1, 1, 1],
    ),
    torch.ops.aten.rms_norm: lambda: (torch.randn(2, 3, 4), [4], torch.ones(4), 1e-5),
    torch.ops.aten.rsqrt: lambda: (torch.randn(2, 3).abs(),),
    torch.ops.aten.scatter: lambda: (
        torch.randn(2, 3),
        1,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
        torch.randn(2, 3),
    ),
    torch.ops.aten.scatter_add: lambda: (
        torch.randn(2, 3),
        1,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
        torch.randn(2, 3),
    ),
    torch.ops.aten.scatter_reduce: lambda: (
        torch.randn(2, 3),
        1,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
        torch.randn(2, 3),
        "sum",
    ),
    torch.ops.aten.select: lambda: (torch.randn(2, 3, 4), 1, 1),
    torch.ops.aten.sigmoid: lambda: (torch.randn(2, 3),),
    torch.ops.aten.slice: lambda: (torch.randn(2, 3, 4), 1, 0, 2, 1),
    torch.ops.aten.split: lambda: (torch.randn(4, 3), 2, 0),
    torch.ops.aten.split_with_sizes: lambda: (torch.randn(6, 3), [2, 1, 3], 0),
    torch.ops.aten.split_with_sizes_copy: lambda: (torch.randn(6, 3), [2, 1, 3], 0),
    torch.ops.aten.sqrt: lambda: (torch.randn(2, 3).abs(),),
    torch.ops.aten.stack: lambda: ([torch.randn(2, 3), torch.randn(2, 3)], 0),
    torch.ops.aten.sub: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.subtract: lambda: (torch.randn(2, 3), torch.randn(2, 3)),
    torch.ops.aten.sum: lambda: (torch.randn(2, 3, 4),),
    torch.ops.aten.tanh: lambda: (torch.randn(2, 3),),
    torch.ops.aten.unbind: lambda: (torch.randn(3, 2, 4), 0),
    torch.ops.aten.upsample_bicubic2d: lambda: (
        torch.randn(1, 3, 4, 4),
        [8, 8],
        True,
        None,
        None,
    ),
    torch.ops.aten.upsample_linear1d: lambda: (torch.randn(1, 3, 4), [8], True, None),
    torch.ops.aten.vdot: lambda: (torch.randn(5), torch.randn(5)),
    torch.ops.aten.where: lambda: (
        torch.randn(2, 3) > 0,
        torch.randn(2, 3),
        torch.randn(2, 3),
    ),
}

assert sample_inputs_factory.keys() >= set(_qnn_friendly_aten_ops())


@pytest.mark.parametrize("op", sample_inputs_factory.keys())
def test_qnn_ir_alignment(tmp_path: Path, op: torch._ops.OpOverloadPacket):
    """
    When: Convert AIMET ExportedProgram into ONNX
    Then:
      1. ONNX grpah should only consist of a curated subset of ONNX operators
         that are known to be 1-to-1 mappable to QNN IR operators.
      2. "QNN-friendly" ATen operators should be converted to a single ONNX operator.
    """

    class Model(torch.nn.Module):
        def forward(self, *args):
            return op(*args)

    sample_inputs = sample_inputs_factory[op]()

    ep = torch.export.export(
        Model(),
        sample_inputs,
    )

    ep = AimetExportedProgram.from_torch_exported_program(ep)

    expected = Model()(*sample_inputs)
    out = ep.module()(*sample_inputs)

    for out_, expected_ in zip(tree_iter(out), tree_iter(expected)):
        assert torch.allclose(out_, expected_, rtol=1e-3)

    torch.onnx.export(
        ep.module(),
        sample_inputs,
        tmp_path / f"{op.__name__}.onnx",
        opset_version=21,
    )
    onnx_model = onnx.load(tmp_path / f"{op.__name__}.onnx")
    op_types = [node.op_type for node in onnx_model.graph.node]

    non_grid_preserving_ops = []
    for op_type in op_types:
        assert op_type in _onnx_to_qnn_ir or _is_grid_preserving_op(op_type)
        if not _is_grid_preserving_op(op_type) and not _is_metadata_op(op_type):
            non_grid_preserving_ops.append(op_type)

    if op in _qnn_friendly_aten_ops():
        assert len(non_grid_preserving_ops) <= 1
