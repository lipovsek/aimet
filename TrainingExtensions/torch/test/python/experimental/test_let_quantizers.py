# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Test Let modules"""

import pytest
import torch
import copy
from torch import nn

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from aimet_torch.experimental.omniquant._utils import (
    replace_with_omniquant_weight_quantizers,
)

from aimet_torch.v2.nn.transformers.models.llama.modeling_llama import (
    QuantizedLlamaRMSNorm,
)
from aimet_torch.v2.nn.transformers.models.gemma3.modeling_gemma3 import (
    QuantizedGemma3RMSNorm,
)
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel


def fold_test(sim: QuantizationSimModel):
    """
    Test fold
    On folding the LET scale to weights we update the original model weights
    l1.w = w/s
    l2.w = w*s
    """
    for _, module in enumerate(sim.model):
        orig_wt = module.weight.cpu().detach().clone()
        for key, quantizer in module.param_quantizers.items():
            if quantizer is not None:
                quantizer.fold_let_params(
                    module, key
                )  # Fold the scale into the weights
        scale_folded_wts = module.weight.cpu().detach()

        if isinstance(module, QuantizedGemma3RMSNorm):
            prev_scale = module.param_quantizers["weight"]._cached_prev_scale
            orig_wt = (orig_wt / prev_scale) + (1 / prev_scale) - 1

            assert torch.equal(orig_wt, scale_folded_wts)
        else:
            prev_scale = module.param_quantizers["weight"]._cached_prev_scale
            foll_scale = module.param_quantizers["weight"]._cached_foll_scale
            factor = prev_scale if prev_scale is not None else 1 / foll_scale

            assert torch.equal(orig_wt, scale_folded_wts * factor)


def get_conv_conv(bias):
    """
    2 layer sequential model for conv conv pair test
    """

    def conv_conv():
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        model = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        ).eval()
        inp = torch.rand(1, input_dim, 32, 32)
        return model, inp

    return conv_conv


def get_lin_lin(bias):
    """
    2 layer sequential model for linear linear pair test
    """

    def lin_lin():
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        model = nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, bias=bias),
            torch.nn.Linear(hidden_dim, output_dim, bias=bias),
        ).eval()
        inp = torch.rand(1, input_dim)
        return model, inp

    return lin_lin


def get_norm_lin(NormLayer):
    """
    2 layer sequential model for Norm and Linear pair test
    """

    def norm_lin():
        input_dim = 3
        output_dim = 2
        model = nn.Sequential(
            NormLayer(input_dim),
            nn.Linear(input_dim, output_dim),
        ).eval()

        inp = torch.rand(1, input_dim)
        return model, inp

    return norm_lin


def update_ref_model(sim: QuantizationSimModel, prev_scale, foll_scale):
    """
    :param sim: QuantizationSimModel
    :param prev_scale: prev_scale set to the QuantizationSimModel model with let_quantized modules
    :param foll_scale: foll_scale set to the QuantizationSimModel model with let_quantized modules
    This sim let-quantized model has prev and foll scale set to None

    Apply prev_scale and foll_scale to sim manually
    Clamp the scale multiplied weight to the param_quantizers max already recorded by compute encodings
    Copy the scale multiplied wts and bias to sim model

    compute_encodings is not computed for weight update during the LET blockwise training,
    So the min, max stays the same during the training.
    Clamping the scale multiplied weight to the recorded max will reflect a similiar behaviour that is expected in omniquant
    """
    assert len(sim.model) == 2, (
        "Only 2 layer sequential model is supported for LETModule unit test"
    )

    for idx, module in enumerate(sim.model):
        wt = module.weight

        if isinstance(module, QuantizedGemma3RMSNorm):
            wt_s = (wt / prev_scale) + (1 / prev_scale) - 1
            sim.model[idx].weight.copy_(wt_s)
            continue

        wt_s = wt * ((1 / prev_scale) if idx == 0 else foll_scale)
        try:
            _max = getattr(module.param_quantizers["weight"], "max")
            wt_s = torch.clamp(wt_s, max=_max)
        except:
            pass

        with torch.no_grad():
            sim.model[idx].weight.copy_(wt_s)

        if isinstance(module, QuantizedLlamaRMSNorm) or module.bias is None:
            continue

        bias_s = module.bias
        if idx == 0:
            bias_s = bias_s * (1 / prev_scale)

        with torch.no_grad():
            module.bias.copy_(bias_s)


@pytest.mark.parametrize(
    "inp_fn",
    [
        get_lin_lin(True),
        get_lin_lin(False),
        get_conv_conv(True),
        get_conv_conv(False),
        get_norm_lin(nn.LayerNorm),
        get_norm_lin(LlamaRMSNorm),
        get_norm_lin(Gemma3RMSNorm),
    ],
)
def test_pair(inp_fn):
    """
    Test the LET modules and LET pairs:
        1. Let modules are getting replaced as expected in QuantizationSimModel
        2. Scales are applied correctly LET modules
        3. Scales are folded back to LET modules
    """
    model, inp = inp_fn()

    out_fp = model(inp)

    sim = QuantizationSimModel(model, inp, config_file="htp_v81")

    # Disable activation quantizers
    # pylint: disable=protected-access
    for _, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_activation_quantizers()

    sim.compute_encodings(lambda model, _: model(inp), None)
    sim_out = sim.model(inp)  # Quantized toy model

    replace_with_omniquant_weight_quantizers(sim.model)

    # forward pass through toy model with let_quantized module
    sim_out_with_no_scale = sim.model(inp)

    # sim_out_with_no_scale  and sim_out is expected to be similar.
    # No scale has been set, hence no modifications to params
    assert torch.equal(sim_out, sim_out_with_no_scale)

    # Copy of let_quantized model with no scale
    # This is used as a reference to compare the output for a let-quantized model with
    # non-zero prev and foll
    ref_let_sim_model = copy.deepcopy(sim)

    sim.compute_encodings(lambda model, _: model(inp), None)

    # Setting different prev and foll scale to test if all params/quantizers are getting updated
    prev_scale = torch.nn.Parameter(torch.tensor([2], dtype=torch.float32))
    foll_scale = torch.nn.Parameter(torch.tensor([20], dtype=torch.float32))

    sim.model[0].param_quantizers["weight"].register_let_params(prev_scale=prev_scale)
    if getattr(sim.model[0], "bias", None) is not None:
        sim.model[0].param_quantizers["bias"].register_let_params(prev_scale=prev_scale)
    sim.model[1].param_quantizers["weight"].register_let_params(foll_scale=foll_scale)

    out_with_radn_scale = sim.model(inp)

    # Model params are updated due to non zero scale.
    # Prev and foll scale are different, hence sim_out, out_with_radn_scale are expected to be diferent
    assert not torch.allclose(sim_out, out_with_radn_scale, atol=0.01)

    sim.compute_encodings(lambda model, _: model(inp), None)

    # Set scale to 2
    prev_scale = torch.nn.Parameter(torch.tensor([2], dtype=torch.float32))
    foll_scale = torch.nn.Parameter(torch.tensor([2], dtype=torch.float32))

    sim.model[0].param_quantizers["weight"].register_let_params(prev_scale=prev_scale)
    if getattr(sim.model[0], "bias", None) is not None:
        sim.model[0].param_quantizers["bias"].register_let_params(prev_scale=prev_scale)
    sim.model[1].param_quantizers["weight"].register_let_params(foll_scale=foll_scale)
    with torch.no_grad():
        out_with_scale_2 = sim.model(inp)

    # Reference model with updated wts and bias with prev_scale & foll_scale
    with torch.no_grad():
        update_ref_model(ref_let_sim_model, prev_scale, foll_scale)
        ref_out = ref_let_sim_model.model(inp)

    # Output of out_with_scale_2 and ref_out should match
    assert torch.allclose(out_with_scale_2, ref_out, atol=1e-05)

    # Test for folding scales into weight before removing quantizers.
    fold_test(sim)

    # pylint: disable=protected-access
    # remove all qunatizers
    for _, module in sim.model.named_modules():
        if isinstance(module, QuantizationMixin):
            module._remove_all_quantizers()

    out_with_quantizers_disabled = sim.model(inp)
    # out_with_quantizers_disabled and out_fp should be same as quantizers were disabled
    assert torch.equal(out_fp, out_with_quantizers_disabled)
