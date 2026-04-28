# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import onnx
import torch
import torch.nn as nn

from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
    fix_node_names_pass,
)

import pytest

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class DummyLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x + residual
        return x


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DummyLayer(3, 16)
        self.layer2 = DummyLayer(16, 32)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SequentialLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            SequentialLayer(3, 16),
            SequentialLayer(16, 32),
        )

    def forward(self, x):
        return self.features(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_and_fix(torch_model, dummy_input, tmp_path, tag):
    torch_model.eval()

    torch.onnx.export(
        torch_model,
        (dummy_input,),
        str(tmp_path / f"{tag}_torchscript.onnx"),
        dynamo=False,
    )
    ts_model = onnx.load(str(tmp_path / f"{tag}_torchscript.onnx"))

    onnx_program = torch.onnx.export(
        torch_model,
        (dummy_input,),
        dynamo=True,
        optimize=False,
    )
    model = onnx_program.model_proto
    onnx.save(model, str(tmp_path / f"{tag}_dynamo.onnx"))

    fixed_model = fix_node_names_pass(model)
    onnx.save(fixed_model, str(tmp_path / f"{tag}_dynamo_fixed.onnx"))

    return ts_model, fixed_model


def _assert_ts_names_in_fixed(ts_model, fixed_model, label):
    ts_names = {node.name for node in ts_model.graph.node}
    fixed_names = {node.name for node in fixed_model.graph.node}

    missing = ts_names - fixed_names
    assert not missing, (
        f"[{label}] Expected torchscript names not found in fixed model: {missing}"
    )

    model_output_names = {o.name for o in fixed_model.graph.output}
    for node in fixed_model.graph.node:
        for i, out in enumerate(node.output):
            if out in model_output_names:
                continue
            expected = f"{node.name}_output_{i}"
            assert out == expected, (
                f"[{label}] Node {node.name} output {i}: "
                f"got {out!r}, expected {expected!r}"
            )

    onnx.checker.check_model(fixed_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skip_on_windows_amd64("Dynamo export fails on amd64")
@pytest.mark.skip_on_windows_arm64("Dynamo export fails on arm64")
def test_fix_node_names_residual_model(tmp_path):
    torch_model = DummyModel()
    dummy_input = torch.randn(1, 3, 224, 224)

    ts_model, fixed_model = _export_and_fix(
        torch_model, dummy_input, tmp_path, "residual"
    )
    _assert_ts_names_in_fixed(ts_model, fixed_model, "DummyModel")


@pytest.mark.skip_on_windows_amd64("Dynamo export fails on amd64")
@pytest.mark.skip_on_windows_arm64("Dynamo export fails on arm64")
def test_fix_node_names_sequential(tmp_path):
    torch_model = SequentialModel()
    dummy_input = torch.randn(1, 3, 224, 224)

    ts_model, fixed_model = _export_and_fix(
        torch_model, dummy_input, tmp_path, "sequential"
    )
    _assert_ts_names_in_fixed(ts_model, fixed_model, "SequentialModel")
