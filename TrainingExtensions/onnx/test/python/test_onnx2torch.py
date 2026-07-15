# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import onnx_ir
import onnxruntime as ort
import pytest
import torch
from torch import nn as nn

from aimet_onnx.experimental.adascale.model_converter import get_pt_block

# Importing onnx2torch_ext (done transitively by model_converter) registers the
# custom ScatterElements converter on the onnx2torch registry.
from .utils import tmp_dir  # noqa: F401  pytest fixture


class ScatterElementsModel(nn.Module):
    """Emits an ONNX ScatterElements node for the given reduce mode.

    ``reduce`` is a torch.scatter_reduce reduce name ("sum"/"prod"/"amax"/"amin")
    or ``None`` for a plain (overwrite) scatter. torch.onnx.export lowers these to
    a ScatterElements node whose ONNX ``reduction`` attribute the converter reads.
    """

    def __init__(self, axis: int, reduce: str | None):
        super().__init__()
        self.axis = axis
        self.reduce = reduce

    def forward(self, data, indices, updates):
        if self.reduce is None:
            return torch.scatter(data, self.axis, indices, updates)
        return data.scatter_reduce(
            self.axis,
            indices,
            updates,
            reduce=self.reduce,
            include_self=True,
        )


def _export_and_convert(model, inputs, tmp_dir, name, input_names):
    model.eval()
    onnx_path = os.path.join(tmp_dir, f"{name}.onnx")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=["output"],
        dynamo=True,
        verbose=False,
    )
    onnx_model = onnx_ir.load(onnx_path)
    pt_block, _ = get_pt_block(onnx_model, (input_names, ["output"]))
    return onnx_path, pt_block


def _run_ort(onnx_path, feed):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {name: t.numpy() for name, t in feed.items()})[0]


class ReduceL2Model(nn.Module):
    """Emits an ONNX ReduceL2 node (L2 norm over ``dim``)."""

    def __init__(self, dim, keepdim: bool):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, data):
        return torch.linalg.vector_norm(data, ord=2, dim=self.dim, keepdim=self.keepdim)


@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_reduce_l2(dim, keepdim, tmp_dir):
    """Three-way equivalence for ReduceL2 across dim x keepdim."""

    data = torch.randn(2, 3, 4)

    model = ReduceL2Model(dim=dim, keepdim=keepdim)
    onnx_path, pt_block = _export_and_convert(
        model, (data,), tmp_dir, f"reduce_l2_dim{dim}_keep{keepdim}", ["data"]
    )

    # 1) original torch model, 2) ONNX Runtime, 3) onnx2torch-converted block.
    orig_out = model(data).detach().numpy()
    onnx_out = _run_ort(onnx_path, {"data": data})
    converted_out = pt_block(data).detach().numpy()

    np.testing.assert_allclose(orig_out, onnx_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(orig_out, converted_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reduce", [None, "sum", "prod", "amax", "amin"])
def test_scatter_elements(axis, reduce, tmp_dir):
    """Three-way equivalence across every axis x reduce mode.

    The original torch model, ONNX Runtime on the exported graph, and the
    onnx2torch-converted block must all agree. Duplicate indices along the
    scattered axis ensure the reduction actually folds multiple updates.
    """
    # (3, 3) data; indices duplicate along `axis` so the reduction actually folds.
    data = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    if axis == 0:
        indices = torch.tensor([[0, 0, 1], [0, 2, 1], [2, 2, 0]])
    else:
        indices = torch.tensor([[0, 0, 1], [2, 2, 0], [1, 1, 2]])
    updates = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])

    model = ScatterElementsModel(axis=axis, reduce=reduce)
    onnx_path, pt_block = _export_and_convert(
        model,
        (data, indices, updates),
        tmp_dir,
        f"scatter_ax{axis}_{reduce}",
        ["data", "indices", "updates"],
    )

    # 1) original torch model, 2) ONNX Runtime, 3) onnx2torch-converted block.
    orig_out = model(data, indices, updates).detach().numpy()
    onnx_out = _run_ort(
        onnx_path, {"data": data, "indices": indices, "updates": updates}
    )
    converted_out = pt_block(data, indices, updates).detach().numpy()

    np.testing.assert_array_equal(orig_out, onnx_out)
    np.testing.assert_array_equal(orig_out, converted_out)
