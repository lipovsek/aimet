# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path
from packaging import version
import torch
from torchvision.models import resnet18, mobilenet_v3_large
import torch.nn.functional as F
from torch.export import ExportedProgram
from aimet_torch import QuantizationSimModel
from aimet_torch.nn import QuantizationMixin
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
        mobilenet_v3_large,
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


def test_dynamo_error():
    class CustomModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(10))

        def forward(self, x):
            return F.linear(x, self.weight)

    @QuantizationMixin.implements(CustomModule)
    class QuantizedCustomModule(QuantizationMixin, CustomModule):
        def forward(self, x):
            # Quantize input tensors
            if self.input_quantizers[0]:
                x = self.input_quantizers[0](x)

            # Run forward with quantized inputs and parameters
            with self._patch_quantized_parameters():
                ret = super().forward(x)

            # Quantize output tensors
            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)

            return ret

    """
    When: Call export with a non-exportable module
    Then: Throw runtime error
    """
    model = torch.nn.Sequential(CustomModule())
    x = torch.randn(10, 10)
    sim = QuantizationSimModel(model, x, config_file="htp_v81")

    with pytest.raises(RuntimeError):
        _ = export(sim.model, args=(x,))
