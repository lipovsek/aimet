# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path
from packaging import version
import torch
import torch.nn.functional as F
from torchvision.models import MobileNetV3, ResNet, resnet18, mobilenet_v3_large
from torch.export import ExportedProgram
from aimet_torch import QuantizationSimModel
from aimet_torch.nn import QuantizationMixin
from aimet_torch.v2.experimental.export import export
import onnx
import pytest


def conv(**_):
    return torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3))


def conv_relu(**_):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.ReLU(),
    )


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.7.0"),
    reason="aimet_torch.export.export is only supported in torch >= 2.7.0",
)
@pytest.mark.parametrize(
    "model_factory",
    [
        conv,
        conv_relu,
        resnet18,
        mobilenet_v3_large,
    ],
)
def test_export(model_factory, tmp_path: Path):
    model = model_factory(pretrained=False).requires_grad_(False).eval()
    x = torch.randn(1, 3, 224, 224)
    sim = QuantizationSimModel(model, x, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(x))

    if isinstance(model, ResNet):
        last_layer = sim.model.fc
    elif isinstance(model, MobileNetV3):
        last_layer = sim.model.classifier[-1]
    else:
        last_layer = sim.model[-1]

    """
    When: Export sim with aimet_torch.export.export
    Then: The resulting ExportedProgram should produce output close enough to sim
    """
    ep: ExportedProgram = export(sim.model, args=(x,))
    path = tmp_path / f"{model_factory.__name__}_quantized.pt2"
    torch.export.save(ep, path)
    ep = torch.export.load(path)
    sim_out = sim.model(x)
    ep_out = ep.module()(x)

    # Allow off-by-3 error
    atol = last_layer.output_quantizers[0].get_scale().item() * 3
    assert torch.allclose(sim_out, ep_out, atol=atol)

    """
    Then: The number of fake_quantize nodes should be equal to that of torch.onnx.export
    """
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

    """
    Then: All scales and zero_points should be constant-folded
    """
    for fake_quantize_node in aten_fake_quantize_nodes:
        for inp_node in fake_quantize_node.all_input_nodes[1:]:
            assert inp_node.op == "placeholder"
            assert inp_node.name.endswith("scale") or inp_node.name.endswith(
                "zero_point"
            )


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.7.0"),
    reason="aimet_torch.export.export is only supported in torch >= 2.7.0",
)
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
