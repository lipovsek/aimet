# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path
from packaging import version
import torch
from torchvision.models import resnet18, mobilenet_v3_large
from torch.export import ExportedProgram
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.experimental.export import export
import onnx
import pytest


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.7.0"),
    reason="aimet_torch.export.export is only supported in torch >= 2.7.0",
)
@pytest.mark.parametrize(
    "model_factory",
    [
        resnet18,
        # mobilenet_v3_large,
    ],
)
def test_export(model_factory, tmp_path: Path):
    model = model_factory(pretrained=False).requires_grad_(False).eval()
    x = torch.randn(1, 3, 224, 224)
    sim = QuantizationSimModel(model, x, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(x))

    """
    When: Export sim with aimet_torch.export.export
    Then:
      1. The resulting ExportedProgram should produce output close enough to sim
      2. The number of fake_quantize nodes should be equal to that of torch.onnx.export
    """
    ep: ExportedProgram = export(sim.model, args=(x,))
    sim_out = sim.model(x)
    ep_out = ep.module()(x)

    if model_factory == resnet18:
        last_layer = sim.model.fc
    else:
        last_layer = sim.model.classifier[-1]

    # Allow off-by-3 error
    atol = last_layer.output_quantizers[0].get_scale().item() * 3
    assert torch.allclose(sim_out, ep_out, atol=atol)

    with torch.no_grad():
        path = tmp_path / f"{model_factory.__name__}_quantized.onnx"
        torch.onnx.export(sim.model, x, path)
        onnx_model = onnx.load_model(path)

    onnx_qdq_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "quantize_dequantize"
    ]
    aten_fake_quantize_nodes = [
        node
        for node in ep.graph.nodes
        if node.op == "call_function"
        and node.target.name().startswith("aten::fake_quantize")
    ]
    assert len(aten_fake_quantize_nodes) == len(onnx_qdq_nodes)
