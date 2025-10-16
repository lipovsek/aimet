# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
import itertools
from pathlib import Path
from packaging import version
import torch
from torchvision.models import resnet18, mobilenet_v3_large
from torch.export import ExportedProgram
import aimet_torch
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.experimental.export import export
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import aimet_torch.v2.nn.transformers.models.llama
from aimet_torch.quantization.affine import AffineQuantizerBase
from aimet_torch.model_preparer import prepare_model
import pytest


def conv(**_):
    return torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3))


def conv_relu(**_):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.ReLU(),
    )


def custom_rmsnorm(**_):
    return torch.nn.Sequential(
        aimet_torch.nn.modules.custom.RmsNorm([1, 3, 224, 224], [-1], epsilon=1e-6)
    )


def llama_rms_norm(**_):
    return torch.nn.Sequential(
        LlamaRMSNorm(hidden_size=224),
    )


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.8.0"),
    reason="aimet_torch.export.export is only supported in torch >= 2.8.0",
)
@pytest.mark.parametrize(
    "model_factory",
    [
        conv,
        conv_relu,
        custom_rmsnorm,
        llama_rms_norm,
        resnet18,
        mobilenet_v3_large,
    ],
)
def test_export(model_factory, tmp_path: Path):
    model = model_factory(pretrained=False).requires_grad_(False).eval()
    model = prepare_model(model)
    x = torch.randn(1, 3, 224, 224)
    sim = QuantizationSimModel(model, x, config_file="htp_v81")

    with pytest.raises(RuntimeError):
        # Before computing encodings, export should raise error
        export(sim.model, args=(x,))

    sim.compute_encodings(lambda model: model(x))

    last_layer_name = [name for name, _ in model.named_modules()][-1]
    last_layer = sim.model.get_submodule(last_layer_name)

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
    Then: The number of qdq nodes should be equal to the number of quantizers in sim
    """
    quantizers = [
        qtzr
        for qmodule in sim.qmodules()
        for qtzr in itertools.chain(
            qmodule.input_quantizers,
            qmodule.output_quantizers,
            qmodule.param_quantizers.values(),
        )
        if isinstance(qtzr, AffineQuantizerBase) and qtzr.is_initialized()
    ]
    torch_dq_nodes = [
        node
        for node in ep.graph.nodes
        if node.op == "call_function"
        and node.target.name().startswith("quantized_decomposed::dequantize")
    ]
    # Exported model can contain MORE qdq nodes than quantsim
    # as data movement op's output encodings that are generated
    # on-the-fly during export
    assert len(torch_dq_nodes) >= len(quantizers)

    """
    Then: All scales and zero_points should be constant-folded
    """
    for q_dq_node in ep.graph.nodes:
        if q_dq_node.op == "call_function" and (
            q_dq_node.target.name().startswith("quantized_decomposed::quantize")
            or q_dq_node.target.name().startswith("quantized_decomposed::dequantize")
        ):
            for inp_node in q_dq_node.all_input_nodes[1:]:
                assert inp_node.op == "placeholder"
                assert inp_node.name.endswith("scale") or inp_node.name.endswith(
                    "zero_point"
                )

    """
    Then: There should be no dangling node, graph_signature or state dict entry
    """
    stack = [ep.graph.output_node()]
    visited = set()
    while stack:
        node = stack.pop(-1)
        if node in visited:
            continue
        visited.add(node.name)
        stack += node.all_input_nodes
    assert visited == set(node.name for node in ep.graph.nodes)

    from torch.export.graph_signature import InputKind

    for input_spec in ep.graph_signature.input_specs:
        assert input_spec.arg.name in visited

        if input_spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
        ):
            assert (
                input_spec.target in ep.state_dict.keys()
                or input_spec.target in ep.constants.keys()
            )

    all_targets = set(
        input_spec.target for input_spec in ep.graph_signature.input_specs
    )
    assert not (ep.state_dict.keys() - all_targets)
    assert not (ep.constants.keys() - all_targets)
