# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
import contextlib
import itertools
import tempfile
import os
import json
import pytest
import random
import numpy as np
import pathlib

import onnx
import torch
from torch import nn, randn

from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_common.defs import QuantizationDataType, QuantScheme
import aimet_torch
from aimet_torch import onnx_utils
from aimet_torch.v2.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.v2.quantization import DequantizedTensor
from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import (
    AffineQuantizerBase,
    GroupedBlockQuantizeDequantize,
    QuantizeDequantize,
)
from aimet_torch.v2.experimental import propagate_output_encodings
from aimet_torch.v2.nn import (
    BaseQuantizationMixin,
    QuantizationMixin,
    QuantizedConv2d,
    QuantizedLinear,
    QuantizedReLU,
)
import aimet_torch.v2.nn.modules.custom as custom
from ..models_ import test_models


def encodings_are_close(
    quantizer_1: AffineQuantizerBase, quantizer_2: AffineQuantizerBase
):
    min_1, max_1 = quantizer_1.get_min(), quantizer_1.get_max()
    min_2, max_2 = quantizer_2.get_min(), quantizer_2.get_max()
    return (
        torch.allclose(min_1, min_2)
        and torch.allclose(max_1, max_2)
        and quantizer_1.bitwidth == quantizer_2.bitwidth
        and quantizer_1.symmetric == quantizer_2.symmetric
    )


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)


@contextlib.contextmanager
def set_export_to_onnx_direct(export_to_onnx_direct):
    entry_state = onnx_utils.EXPORT_TO_ONNX_DIRECT
    onnx_utils.EXPORT_TO_ONNX_DIRECT = export_to_onnx_direct
    yield
    onnx_utils.EXPORT_TO_ONNX_DIRECT = entry_state


class ConcatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = custom.Concat()

    def forward(self, *x):
        return self.cat(*x)


class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(2, 2))

    def forward(self, x):
        return self.conv(x)


class TestQuantsim:
    """Test Percentile quantization scheme"""

    @pytest.mark.parametrize(
        "weight_bw, act_bw, is_valid",
        [
            (16, 16, True),
            (4, 4, True),
            (2, 32, True),
            (4, 2, False),
            (1, 8, False),
            (4, 32, True),
            (4, 33, False),
            (32, 4, True),
            (33, 4, False),
        ],
    )
    def test_invalid_bw_instantiation(self, weight_bw, act_bw, is_valid):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        if is_valid:
            qsim = QuantizationSimModel(
                model, dummy_input, default_param_bw=weight_bw, default_output_bw=act_bw
            )
            assert qsim.model.conv.param_quantizers["weight"].bitwidth == weight_bw
            assert qsim.model.relu.output_quantizers[0].bitwidth == act_bw
        else:
            with pytest.raises(ValueError):
                qsim = QuantizationSimModel(
                    model,
                    dummy_input,
                    default_param_bw=weight_bw,
                    default_output_bw=act_bw,
                )

    def test_set_percentile_value(self):
        """Test pecentile scheme by setting different percentile values"""

        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input, quant_scheme="percentile")
        weight_quantizer = sim.model.conv.param_quantizers["weight"]
        assert isinstance(
            weight_quantizer.encoding_analyzer, PercentileEncodingAnalyzer
        )

        sim.set_percentile_value(99.9)
        assert weight_quantizer.encoding_analyzer.percentile == 99.9

        sim.compute_encodings(forward_pass, None)
        weight_max_99p9 = weight_quantizer.get_max()

        sim.set_percentile_value(90.0)
        assert weight_quantizer.encoding_analyzer.percentile == 90.0
        sim.compute_encodings(forward_pass, None)
        weight_max_90p0 = weight_quantizer.get_max()

        assert torch.all(weight_max_99p9.gt(weight_max_90p0))

    @pytest.mark.parametrize("config_file", (None, get_path_for_per_channel_config()))
    def test_set_and_freeze_param_encodings(self, config_file):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(
            model,
            dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            config_file=config_file,
        )
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = "test_model"
            sim.export(temp_dir, fname, dummy_input)
            file_path = os.path.join(temp_dir, fname + "_torch.encodings")

            sim_2 = QuantizationSimModel(model, dummy_input, config_file=config_file)

            """
            When: call set_and_freeze_param_encodigns
            Then: Encodings should match
            """
            sim_2.set_and_freeze_param_encodings(file_path)
            assert encodings_are_close(
                sim.model.conv.param_quantizers["weight"],
                sim_2.model.conv.param_quantizers["weight"],
            )

        """
        When: Recompute encodings with new weights
        Then: Weight encodings should NOT get overwritten by compute_encodings
        """
        weight_min = sim_2.model.conv.param_quantizers["weight"].min.clone().detach()
        weight_max = sim_2.model.conv.param_quantizers["weight"].max.clone().detach()

        with torch.no_grad():
            sim_2.model.conv.weight.mul_(10)

        sim_2.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.equal(weight_min, sim_2.model.conv.param_quantizers["weight"].min)
        assert torch.equal(weight_max, sim_2.model.conv.param_quantizers["weight"].max)

        """
        When: Recompute encodings with new input
        Then: Activation encodings should be updated for the new input (freezing only takes effect to weight quantizers)
        """
        new_dummy_input = 10 * dummy_input
        input_min = sim_2.model.conv.input_quantizers[0].min.clone().detach()
        input_max = sim_2.model.conv.input_quantizers[0].max.clone().detach()
        sim_2.compute_encodings(lambda model, _: model(new_dummy_input), None)
        assert torch.allclose(input_min * 10, sim_2.model.conv.input_quantizers[0].min)
        assert torch.allclose(input_max * 10, sim_2.model.conv.input_quantizers[0].max)

    @pytest.mark.parametrize("config_file", (None, get_path_for_per_channel_config()))
    def test_load_and_freeze_encodings(self, config_file):
        model = test_models.TinyModel()
        dummy_input = torch.rand(1, 3, 32, 32)
        sim = QuantizationSimModel(
            model,
            dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            config_file=config_file,
        )
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = "test_model"
            sim.export(temp_dir, fname, dummy_input)
            file_path = os.path.join(temp_dir, fname + "_torch.encodings")

            """
            When: Load encodings with ``load_and_freeze_encodings``
            Then: No quantizers should get additionally enabled/disabled
            """
            sim_2 = QuantizationSimModel(
                test_models.TinyModel(), dummy_input, config_file=config_file
            )
            all_quantizers = [
                q for q in sim_2.model.modules() if isinstance(q, QuantizerBase)
            ]
            sim_2.load_and_freeze_encodings(file_path)
            assert all_quantizers == [
                q for q in sim_2.model.modules() if isinstance(q, QuantizerBase)
            ]

        """
        When: Recompute encodings with new weights
        Then: Weight encodings should NOT get overwritten by compute_encodings
        """
        weight_min = sim_2.model.conv1.param_quantizers["weight"].min.clone().detach()
        weight_max = sim_2.model.conv1.param_quantizers["weight"].max.clone().detach()

        with torch.no_grad():
            sim_2.model.conv1.weight.mul_(10)

        sim_2.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.equal(weight_min, sim_2.model.conv1.param_quantizers["weight"].min)
        assert torch.equal(weight_max, sim_2.model.conv1.param_quantizers["weight"].max)

        """
        When: Recompute encodings with new input
        Then: Activation encodings should NOT get overwritten by compute_encodings
        """
        new_dummy_input = 10 * dummy_input
        input_min = sim_2.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim_2.model.conv1.input_quantizers[0].max.clone().detach()
        sim_2.compute_encodings(lambda model, _: model(new_dummy_input), None)
        assert torch.equal(input_min, sim_2.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim_2.model.conv1.input_quantizers[0].max)

    def test_load_and_freeze_with_partial_encodings(self):
        """Test load_and_freeze encoding API with partial_encodings"""
        model = test_models.TinyModel()
        dummy_input = torch.randn(1, 3, 32, 32)

        sample_encoding = {
            "min": -4,
            "max": 4,
            "scale": 0.03,
            "offset": 8,
            "bitwidth": 8,
            "is_symmetric": "False",
            "dtype": "int",
        }

        partial_encodings = {
            "activation_encodings": {"conv1": {"input": {"0": sample_encoding}}},
            "param_encodings": {
                "conv1.weight": [sample_encoding] * model.conv1.out_channels
            },
        }

        sim = QuantizationSimModel(
            model, dummy_input, quant_scheme=QuantScheme.post_training_tf
        )
        all_quantizers = [
            q for q in sim.model.modules() if isinstance(q, QuantizerBase)
        ]
        sim.load_and_freeze_encodings(partial_encodings)

        """
        When: Load partial encodings with ``load_and_freeze_encodings``
        Then: No quantizers should get additionally enabled/disabled
        """
        assert all_quantizers == [
            q for q in sim.model.modules() if isinstance(q, QuantizerBase)
        ]

        """
        When: Recompute encodings with new weights
        Then: Weight encodings imported from the config file should NOT get overwritten by compute_encodings
            2) Weight encodings NOT imported from the config file SHOULD get overwritten by compute_encodings
        """
        conv1_weight_min = (
            sim.model.conv1.param_quantizers["weight"].min.clone().detach()
        )
        conv1_weight_max = (
            sim.model.conv1.param_quantizers["weight"].max.clone().detach()
        )
        with torch.no_grad():
            sim.model.conv1.weight.mul_(10)

        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.equal(
            conv1_weight_min, sim.model.conv1.param_quantizers["weight"].min
        )
        assert torch.equal(
            conv1_weight_max, sim.model.conv1.param_quantizers["weight"].max
        )

        """
        When: Recompute encodings with new weights
        Then: Weight encodings NOT imported from the config file SHOULD get overwritten by compute_encodings
        """
        fc_weight_min = sim.model.fc.param_quantizers["weight"].min.clone().detach()
        fc_weight_max = sim.model.fc.param_quantizers["weight"].max.clone().detach()
        with torch.no_grad():
            sim.model.fc.weight.mul_(10)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.allclose(
            fc_weight_min * 10, sim.model.fc.param_quantizers["weight"].min
        )
        assert torch.allclose(
            fc_weight_max * 10, sim.model.fc.param_quantizers["weight"].max
        )

        """
        When: Recompute encodings with new input
        Then: Activation encodings should NOT get overwritten by compute_encodings
            1) Activation encodings imported from the config file should NOT get overwritten by compute_encodings
            2) Activation encodings NOT imported from the config file SHOULD get overwritten by compute_encodings
        """
        new_dummy_input = 10 * dummy_input
        conv1_input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        conv1_input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()
        fc_output_min = sim.model.fc.output_quantizers[0].min.clone().detach()
        fc_output_max = sim.model.fc.output_quantizers[0].max.clone().detach()
        sim.compute_encodings(lambda model, _: model(new_dummy_input), None)
        assert torch.equal(conv1_input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(conv1_input_max, sim.model.conv1.input_quantizers[0].max)
        assert not torch.isclose(fc_output_min, sim.model.fc.output_quantizers[0].min)
        assert not torch.isclose(fc_output_max, sim.model.fc.output_quantizers[0].max)

    def test_load_encodings(self):
        model = test_models.TinyModel()
        dummy_input = torch.randn(1, 3, 32, 32)

        sample_encoding = {
            "min": -4,
            "max": 4,
            "scale": 0.03,
            "offset": 8,
            "bitwidth": 8,
            "is_symmetric": "False",
            "dtype": "int",
        }
        sample_encoding2 = {
            "min": -8,
            "max": 8,
            "scale": 0.06,
            "offset": 8,
            "bitwidth": 8,
            "is_symmetric": "False",
            "dtype": "int",
        }

        encodings = {
            "activation_encodings": {"conv1": {"input": {"0": sample_encoding}}},
            "param_encodings": {
                "conv1.weight": [sample_encoding] * model.conv1.out_channels
            },
        }
        encodings2 = {
            "activation_encodings": {"conv1": {"input": {"0": sample_encoding2}}},
            "param_encodings": {
                "conv1.weight": [sample_encoding2] * model.conv1.out_channels
            },
        }
        encodings3 = {
            "activation_encodings": {
                "conv1": {
                    "input": {"0": sample_encoding},
                    "output": {"0": sample_encoding},
                }
            },
            "param_encodings": {
                "conv1.weight": [sample_encoding] * model.conv1.out_channels
            },
        }

        """
        When: Call load_encodings with strict=True
        Then: Runtime error is raised
        """
        sim = QuantizationSimModel(model, dummy_input)
        with pytest.raises(RuntimeError):
            sim.load_encodings(encodings3, strict=True)

        """
        When: Call load_encodings with strict=False
        Then: Skip to load encodings that doesn't exist
        """
        sim = QuantizationSimModel(model, dummy_input)
        sim.load_encodings(encodings3, strict=False)
        assert sim.model.conv1.output_quantizers[0] is None

        """
        When: Call load_encodings with partial=False
        Then: All the dangling quantizers should be removed
        """
        sim = QuantizationSimModel(model, dummy_input)
        sim.load_encodings(encodings, partial=False)
        all_quantizers = [
            q for q in sim.model.modules() if isinstance(q, QuantizerBase)
        ]
        assert all_quantizers == [
            sim.model.conv1.param_quantizers["weight"],
            sim.model.conv1.input_quantizers[0],
        ]

        """
        When: Call load_encodings with partial=True
        Then: No quantizer gets removed
        """
        sim = QuantizationSimModel(model, dummy_input)
        all_quantizers = [
            q for q in sim.model.modules() if isinstance(q, QuantizerBase)
        ]
        sim.load_encodings(encodings, partial=True)
        assert all_quantizers == [
            q for q in sim.model.modules() if isinstance(q, QuantizerBase)
        ]

        for requires_grad in (True, False):
            """
            When: Call load_encodings with requires_grad specified
            Then: The loaded quantizers should be set to requires_grad=True/False accordingly
            """
            sim = QuantizationSimModel(model, dummy_input)
            all_parameters = {
                q: (q.min.clone(), q.max.clone())
                for q in sim.model.modules()
                if isinstance(q, QuantizerBase)
            }
            sim.load_encodings(encodings, requires_grad=requires_grad)
            assert (
                sim.model.conv1.param_quantizers["weight"].min.requires_grad
                == sim.model.conv1.param_quantizers["weight"].max.requires_grad
                == requires_grad
            )
            assert (
                sim.model.conv1.input_quantizers[0].min.requires_grad
                == sim.model.conv1.input_quantizers[0].max.requires_grad
                == requires_grad
            )

            # requires_grad of all the oither quantization parameters should not be modified
            for q, (min_copy, max_copy) in all_parameters.items():
                if q in (
                    sim.model.conv1.param_quantizers["weight"],
                    sim.model.conv1.input_quantizers[0],
                ):
                    continue
                assert q.min.requires_grad == min_copy.requires_grad
                assert q.max.requires_grad == max_copy.requires_grad

            """
            When: Call load_encodings with requires_grad NOT specified
            Then: requires_grad flag should be kept unchanged
            """
            sim.load_encodings(encodings, requires_grad=None)
            assert (
                sim.model.conv1.param_quantizers["weight"].min.requires_grad
                == sim.model.conv1.param_quantizers["weight"].max.requires_grad
                == requires_grad
            )
            assert (
                sim.model.conv1.input_quantizers[0].min.requires_grad
                == sim.model.conv1.input_quantizers[0].max.requires_grad
                == requires_grad
            )

            # requires_grad of all the oither quantization parameters should not be modified
            for q, (min_copy, max_copy) in all_parameters.items():
                if q in (
                    sim.model.conv1.param_quantizers["weight"],
                    sim.model.conv1.input_quantizers[0],
                ):
                    continue
                assert q.min.requires_grad == min_copy.requires_grad
                assert q.max.requires_grad == max_copy.requires_grad

        """
        When: Call load_encodings with allow_overwrite=True
        Then: The loaded quantizers should be overwritten by a subsequent
              compute_encodings or load_encodings
        """
        sim = QuantizationSimModel(model, dummy_input)
        sim.load_encodings(encodings, allow_overwrite=True)
        weight_min = sim.model.conv1.param_quantizers["weight"].min.clone().detach()
        weight_max = sim.model.conv1.param_quantizers["weight"].max.clone().detach()
        input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()

        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        assert not torch.allclose(
            weight_min, sim.model.conv1.param_quantizers["weight"].min
        )
        assert not torch.allclose(
            weight_max, sim.model.conv1.param_quantizers["weight"].max
        )
        assert not torch.allclose(input_min, sim.model.conv1.input_quantizers[0].min)
        assert not torch.allclose(input_max, sim.model.conv1.input_quantizers[0].max)

        weight_min = sim.model.conv1.param_quantizers["weight"].min.clone().detach()
        weight_max = sim.model.conv1.param_quantizers["weight"].max.clone().detach()
        input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()

        sim.load_encodings(encodings2)

        assert not torch.allclose(
            weight_min, sim.model.conv1.param_quantizers["weight"].min
        )
        assert not torch.allclose(
            weight_max, sim.model.conv1.param_quantizers["weight"].max
        )
        assert not torch.allclose(input_min, sim.model.conv1.input_quantizers[0].min)
        assert not torch.allclose(input_max, sim.model.conv1.input_quantizers[0].max)

        """
        When: Call load_encodings with allow_overwrite=False
        Then: The loaded quantizers should NOT be overwritten by a subsequent
              compute_encodings or load_encodings
        """
        sim = QuantizationSimModel(model, dummy_input)
        sim.load_encodings(encodings, allow_overwrite=False)
        weight_min = sim.model.conv1.param_quantizers["weight"].min.clone().detach()
        weight_max = sim.model.conv1.param_quantizers["weight"].max.clone().detach()
        input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()

        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        assert torch.equal(weight_min, sim.model.conv1.param_quantizers["weight"].min)
        assert torch.equal(weight_max, sim.model.conv1.param_quantizers["weight"].max)
        assert torch.equal(input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim.model.conv1.input_quantizers[0].max)

        sim.load_encodings(encodings2)

        assert torch.equal(weight_min, sim.model.conv1.param_quantizers["weight"].min)
        assert torch.equal(weight_max, sim.model.conv1.param_quantizers["weight"].max)
        assert torch.equal(input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim.model.conv1.input_quantizers[0].max)

        """
        When: Call load_encodings with allow_overwrite=None
        Then: Whether the loaded quantizers can be overwritten is kept unchanged
        """
        sim.load_encodings(encodings, allow_overwrite=None)

        assert torch.equal(weight_min, sim.model.conv1.param_quantizers["weight"].min)
        assert torch.equal(weight_max, sim.model.conv1.param_quantizers["weight"].max)
        assert torch.equal(input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim.model.conv1.input_quantizers[0].max)

    @pytest.mark.parametrize(
        "load_encodings_fn",
        [
            load_encodings_to_sim,
            QuantizationSimModel.load_and_freeze_encodings,
            QuantizationSimModel.set_and_freeze_param_encodings,
        ],
    )
    def test_legacy_load_encodings_partial_encoding(self, load_encodings_fn):
        model = test_models.SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        partial_torch_encodings = {
            "activation_encodings": {
                "conv1": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138,
                        }
                    }
                },
                "conv2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.4923851788043976,
                            "min": -0.43767568469047546,
                            "offset": -120,
                            "scale": 0.0036472973879426718,
                        }
                    }
                },
                "fc2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.1948324590921402,
                            "min": -0.15752412378787994,
                            "offset": -114,
                            "scale": 0.0013817904982715845,
                        }
                    }
                },
                "relu1": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 1.0608084201812744,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.004160033073276281,
                        }
                    }
                },
                "relu3": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.5247029066085815,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.0020576585084199905,
                        }
                    }
                },
            },
            "excluded_layers": [],
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225,
                    }
                ]
                * model.conv1.out_channels,
                "fc2.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.13095608353614807,
                        "min": -0.14966410398483276,
                        "offset": -8,
                        "scale": 0.018708012998104095,
                    }
                ],
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced",
            },
            "version": "0.6.1",
        }

        qsim = QuantizationSimModel(model, dummy_input)
        quantizers = [q for q in qsim.model.modules() if isinstance(q, QuantizerBase)]

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, "w") as f:
                json.dump(partial_torch_encodings, f)

            load_encodings_fn(qsim, fname)

        if load_encodings_fn is load_encodings_to_sim:
            """
            When: Load partial encodings with load_encodings_to_sim
            Then: Quantizers that have no corresponding encodings should be removed
            """
            loaded_quantizers = [
                qsim.model.conv1.input_quantizers[0],
                qsim.model.conv1.param_quantizers["weight"],
                qsim.model.conv2.output_quantizers[0],
                qsim.model.fc2.output_quantizers[0],
                qsim.model.fc2.param_quantizers["weight"],
                qsim.model.relu1.output_quantizers[0],
                qsim.model.relu3.output_quantizers[0],
            ]
            assert sorted(loaded_quantizers, key=id) == sorted(
                [q for q in qsim.model.modules() if isinstance(q, QuantizerBase)],
                key=id,
            )

        elif load_encodings_fn in [
            QuantizationSimModel.load_and_freeze_encodings,
            QuantizationSimModel.set_and_freeze_param_encodings,
        ]:
            """
            When: Load partial encodings with load_and_freeze_encodings or set_and_freeze_param_encodings
            Then: Quantizers shouldn't be additionally removed or instantiated
            """
            assert quantizers == [
                q for q in qsim.model.modules() if isinstance(q, QuantizerBase)
            ]
        else:
            raise AssertionError

    @pytest.mark.parametrize(
        "load_encodings_fn",
        [
            load_encodings_to_sim,
            QuantizationSimModel.load_and_freeze_encodings,
            QuantizationSimModel.set_and_freeze_param_encodings,
        ],
    )
    def test_legacy_load_encodings_mismatching_encoding(self, load_encodings_fn):
        model = test_models.SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        invalid_torch_encodings = {
            "excluded_layers": [],
            "activation_encodings": {
                "conv999": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138,
                        }
                    }
                },
            },
            "param_encodings": {
                "conv999.weight": [  # NOTE: conv999 does not exist in the model
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225,
                    }
                ],
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced",
            },
            "version": "0.6.1",
        }

        qsim = QuantizationSimModel(model, dummy_input)

        """
        When: Try to load encoding file some keys of which are missing in the model
              (Note that conv999 does not exist in the model)
        Then: Throw runtime error
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, "w") as f:
                json.dump(invalid_torch_encodings, f)

            with pytest.raises(RuntimeError):
                load_encodings_fn(qsim, fname)

    @pytest.mark.parametrize(
        "load_encodings_fn",
        [
            load_encodings_to_sim,
            QuantizationSimModel.load_and_freeze_encodings,
            QuantizationSimModel.set_and_freeze_param_encodings,
        ],
    )
    def test_legacy_load_encodings_to_disabled_quantizer(self, load_encodings_fn):
        model = test_models.SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        invalid_torch_encodings = {
            "excluded_layers": [],
            "activation_encodings": {
                "conv1": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138,
                        }
                    }
                },
            },
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225,
                    }
                ],
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced",
            },
            "version": "0.6.1",
        }

        qsim = QuantizationSimModel(model, dummy_input)

        """
        Given: Input/param quantizers of conv1 is disabled
        When: Try to load input/param quantizers to conv1
        Then: Throw runtime error
        """
        qsim.model.conv1.input_quantizers[0] = None
        qsim.model.conv1.param_quantizers["weight"] = None

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, "w") as f:
                json.dump(invalid_torch_encodings, f)

            with pytest.raises(RuntimeError):
                load_encodings_fn(qsim, fname)

    def test_save_and_load_gbbq(self):
        torch.manual_seed(0)
        model = test_models.SingleResidualWithAvgPool()
        dummy_input = torch.randn(1, 3, 28, 28)
        dummy_input_2 = torch.randn(1, 3, 28, 28)
        qsim = QuantizationSimModel(model, dummy_input)
        qsim.model.fc.param_quantizers["weight"] = GroupedBlockQuantizeDequantize(
            shape=(10, 6),
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=(1, 12),
            block_grouping=(1, 6),
        )
        qsim.compute_encodings(lambda m, _: m(dummy_input), None)
        out1 = qsim.model(dummy_input)
        with tempfile.TemporaryDirectory() as temp_dir:
            qsim.save_encodings_to_json(temp_dir, "saved_encodings")
            qsim.export(temp_dir, "exported_encodings", dummy_input=dummy_input)

            with open(os.path.join(temp_dir, "saved_encodings.json"), "r") as enc_file:
                encodings = json.load(enc_file)

            assert len(encodings["param_encodings"]["fc.weight"]) == 60

            with open(
                os.path.join(temp_dir, "exported_encodings_torch.encodings"), "r"
            ) as enc_file:
                encodings = json.load(enc_file)

            assert len(encodings["param_encodings"]["fc.weight"]) == 60

            old_weight = qsim.model.fc.weight
            old_max = qsim.model.fc.param_quantizers["weight"].get_max()[0][0]
            qsim.model.fc.weight = torch.nn.Parameter(torch.randn(old_weight.shape))
            qsim.compute_encodings(lambda m, _: m(dummy_input_2), None)
            assert qsim.model.fc.param_quantizers["weight"].get_max()[0][0] != old_max
            out2 = qsim.model(dummy_input)

            assert not torch.equal(out1, out2)

            # Test loading of encodings saved using save_encodings_to_json
            qsim.model.fc.weight = old_weight
            qsim.load_encodings(os.path.join(temp_dir, "saved_encodings.json"))

            assert qsim.model.fc.param_quantizers["weight"].get_max()[0][0] == old_max
            out3 = qsim.model(dummy_input)
            assert torch.allclose(out1, out3)

            qsim.model.fc.weight = torch.nn.Parameter(torch.randn(old_weight.shape))
            qsim.compute_encodings(lambda m, _: m(dummy_input_2), None)

            # Test loading of encodings from sim.export
            qsim.model.fc.weight = old_weight
            qsim.load_encodings(
                os.path.join(temp_dir, "exported_encodings_torch.encodings")
            )

            out4 = qsim.model(dummy_input)
            assert torch.allclose(out1, out4)

    def test_quantsim_with_unused_modules(self):
        """
        Given: A model with unused layer
        When: Instantiate quantsim
        Then: 1) No error is not raised
              2) Length of input quantizers is equal to the length defined in __quant_init__
              3) Input quantizers are None
        """

        model = test_models.ModelWithUnusedAdd()
        sim = QuantizationSimModel(model, dummy_input=torch.randn(10, 10))
        assert len(sim.model.add.input_quantizers) == 2
        assert type(sim.model.add.input_quantizers[0]) is type(
            sim.model.add.input_quantizers[1]
        )

        """
        Given: A model with unused layer
        When: Instantiate quantsim
        Then: 1) No error is not raised
              2) Length of output quantizers is equal to the length defined in __quant_init__
              3) Output quantizers are not None
        """
        model = test_models.ModelWithUnusedRNN()
        sim = QuantizationSimModel(model, dummy_input=torch.randn(10, 10))
        assert len(sim.model.rnn.output_quantizers) == 2
        assert type(sim.model.rnn.output_quantizers[0]) is type(
            sim.model.rnn.output_quantizers[1]
        )

    def test_quantsim_with_abstract_modules(self):
        """
        Given: A model with an abstract nn.Module
        When: Instantiate quantsim
        Then: 1) No error is not raised
              2) Abstract modules stay unchanged
              3) If the abstract module contains non-abstract child modules,
                 the child modules should be converted to quantized modules.
        """
        model = test_models.ModelWithAbstractModule()
        sim = QuantizationSimModel(model, dummy_input=torch.randn(1, 3, 10, 10))
        assert type(sim.model.module) == torch.nn.Module
        assert isinstance(sim.model.module.conv, QuantizedConv2d)

    def test_export_concat_encodings(self):
        num_inputs = 3
        model = ConcatModel()
        dummy_input = tuple([torch.randn(1, 3, 32, 32)] * num_inputs)
        sim = QuantizationSimModel(model, dummy_input=dummy_input)
        sim.compute_encodings(lambda model, _: model(*dummy_input), None)
        with tempfile.TemporaryDirectory() as temp_dir:
            fname = "test_model"
            sim.export(temp_dir, fname, dummy_input)
            with open(os.path.join(temp_dir, f"{fname}_torch.encodings")) as f:
                encodings = json.load(f)
            assert (
                len(encodings["activation_encodings"]["cat"]["input"].keys())
                == num_inputs
            )

            sim = QuantizationSimModel(model, dummy_input=dummy_input)
            sim.load_encodings(encodings)
            sim.save_encodings_to_json(temp_dir, "model_encodings")

    @pytest.mark.parametrize("config_file", (None, get_path_for_per_channel_config()))
    def test_expand_op_is_not_quantized(self, config_file):
        model = test_models.ExpandModel()
        sim = QuantizationSimModel(
            model, dummy_input=torch.randn(10), config_file=config_file
        )
        assert sim.model.expand.output_quantizers[0] is None

    def test_encoding_min_max_fixed_vals(self):
        """
        When: Create sim with HTP config file
        Then:
          - Relu output encoding should be partially fixed to [0, ?]
          - Softmax output encoding should be partially fixed to [0, 1]
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()
                self.softmax = torch.nn.Softmax()

            def forward(self, inp):
                x = self.conv(inp)
                x = self.relu(x)
                x = self.softmax(x)
                return x

        model = Model()
        dummy_input = torch.randn(1, 3, 10, 10)

        qsim = QuantizationSimModel(model, dummy_input, config_file="htp_v81")

        assert not qsim.model.relu.output_quantizers[0].is_initialized()
        assert qsim.model.relu.output_quantizers[0].min == 0.0
        assert not qsim.model.relu.output_quantizers[0].min.requires_grad
        assert qsim.model.relu.output_quantizers[0].max.requires_grad

        assert qsim.model.softmax.output_quantizers[0].is_initialized()
        assert qsim.model.softmax.output_quantizers[0].min == 0.0
        assert qsim.model.softmax.output_quantizers[0].max == 1.0
        assert not qsim.model.softmax.output_quantizers[0].min.requires_grad
        assert not qsim.model.softmax.output_quantizers[0].max.requires_grad

        qsim = QuantizationSimModel(
            model,
            dummy_input,
            config_file="htp_v81",
            default_param_bw=16,
            default_output_bw=16,
            default_data_type=QuantizationDataType.float,
        )
        assert not hasattr(qsim.model.relu.output_quantizers[0], "min")
        assert not hasattr(qsim.model.relu.output_quantizers[0], "max")
        assert not hasattr(qsim.model.softmax.output_quantizers[0], "min")
        assert not hasattr(qsim.model.softmax.output_quantizers[0], "max")

    def test_export_to_onnx_direct_fixed_param_names(self):
        torch.manual_seed(0)
        model = test_models.SmallLinearModel()
        dummy_input = torch.randn(1, 8, 3)
        with set_export_to_onnx_direct(True):
            sim = QuantizationSimModel(model, dummy_input)
            sim.compute_encodings(lambda m, _: m(*dummy_input), None)

            with tempfile.TemporaryDirectory() as tmp_dir:
                sim.export(tmp_dir, "single_linear", dummy_input)

                with open(
                    os.path.join(tmp_dir, "single_linear.encodings"), "r"
                ) as encodings_file:
                    encodings = json.load(encodings_file)

                param_encodings_set = {
                    encoding["name"] for encoding in encodings["param_encodings"]
                }

                for name, _ in model.named_parameters():
                    if "bias" not in name:
                        assert name in param_encodings_set

    class CustomLinear(torch.nn.Module):
        """custom linear module"""

        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.matmul = custom.MatMul()
            self.add = custom.Add()

        def forward(self, x):
            x = self.matmul(x, self.weight.transpose(0, 1))
            return self.add(x, self.bias)

    @QuantizationMixin.implements(CustomLinear)
    class QuantizedCustomLinear(QuantizationMixin, CustomLinear):
        def __quant_init__(self):
            super().__quant_init__()
            self.input_quantizers = torch.nn.ModuleList([])
            self.output_quantizers = torch.nn.ModuleList([])

        def forward(self, x):
            with self._patch_quantized_parameters():
                return super().forward(x)

    def test_non_leaf_qmodule(self):
        """
        Given: Define a quantized definition of a non-leaf module
        """

        """
        When: Create quantsim with the non-leaf module
        Then: 1) The non-leaf module should be converted to a quantized module
              2) All its submodules should be also converted to quantized modules
        """
        model = torch.nn.Sequential(
            self.CustomLinear(10, 10),
            torch.nn.Sigmoid(),
        )
        dummy_input = torch.randn(10, 10)

        sim = QuantizationSimModel(model, dummy_input)

        qlinear = sim.model[0]
        assert isinstance(qlinear, self.QuantizedCustomLinear)
        assert isinstance(qlinear.param_quantizers["weight"], AffineQuantizerBase)
        assert qlinear.param_quantizers["bias"] is None

        assert isinstance(qlinear.matmul, custom.QuantizedMatMul)
        assert isinstance(qlinear.matmul.input_quantizers[0], AffineQuantizerBase)
        assert qlinear.matmul.input_quantizers[1] is None
        assert isinstance(qlinear.matmul.output_quantizers[0], AffineQuantizerBase)

        assert isinstance(qlinear.add, custom.QuantizedAdd)
        assert qlinear.add.input_quantizers[0] is None
        assert qlinear.add.input_quantizers[1] is None
        assert isinstance(qlinear.add.output_quantizers[0], AffineQuantizerBase)

        """
        When: Export
        Then: The generated encoding file should contain all entries properly
        """
        sim.compute_encodings(lambda model: model(dummy_input))
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(tmpdir, "model", dummy_input=dummy_input)
            with open(os.path.join(tmpdir, "model_torch.encodings")) as f:
                encodings = json.load(f)

        expected_schema = {
            "activation_encodings": {
                "0.add": {"output": ...},  # CustomLinear.add
                "0.matmul": {"input": ..., "output": ...},  # CustomLinear.matmul
                "1": {"output": ...},  # Sigmoid
            },
            "param_encodings": {
                "0.weight": ...,  # CustomLinear.weight
            },
        }

        def _assert_same_keys(d: dict, expected: dict):
            assert d.keys() == expected.keys()

            for k in d:
                v1, v2 = d[k], expected[k]
                if isinstance(v2, dict):
                    _assert_same_keys(v1, v2)

        _assert_same_keys(
            encodings["activation_encodings"], expected_schema["activation_encodings"]
        )
        # TODO: This assertion currently fails
        # _assert_same_keys(encodings['param_encodings'], expected_schema['param_encodings'])

    def test_non_leaf_qmodule_exception_rules(self):
        quantsim_config = {
            "defaults": {
                "hw_version": "V79",
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "strict_symmetric": "False",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        class SupergroupLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qk_matmul = custom.MatMul()
                self.mask_add = custom.Add()
                self.softmax = torch.nn.Softmax(dim=-1)

            def forward(self, query, key, attn_mask):
                attn_weight = self.qk_matmul(query, key.transpose(-2, -1))
                attn_weight = self.mask_add(attn_weight, attn_mask)
                attn_weight = self.softmax(attn_weight)
                return attn_weight

        @QuantizationMixin.implements(SupergroupLayer)
        class QuantizedSupergroupLayer(QuantizationMixin, SupergroupLayer):
            def __quant_init__(self):
                super().__quant_init__()
                # Supergroup itself doesn't need input/output quantizers
                self.input_quantizers = torch.nn.ModuleList([])
                self.output_quantizers = torch.nn.ModuleList([])

            def forward(self, query, key, attn_mask):
                return super().forward(query, key, attn_mask)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(10, 10)
                self.k = torch.nn.Linear(10, 10)
                self.v = torch.nn.Linear(10, 10)
                self.attn = SupergroupLayer()
                self.matmul = custom.MatMul()

            def forward(self, x, mask):
                attn = self.attn(self.q(x), self.k(x), mask)
                return self.matmul(self.v(x), attn)

        model = Model()
        dummy_input = (torch.randn(1, 10, 10), torch.zeros(1, 1, 10, 10))

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(
                model, dummy_input, default_output_bw=16, config_file=config_path
            )

        sim.compute_encodings(lambda model: model(*dummy_input))
        """
        MatMul second inputs should be symmetric
        """
        assert sim.model.attn.softmax.output_quantizers[0].symmetric
        assert sim.model.k.output_quantizers[0].symmetric

    def test_trivial_leaf_module(self):
        """
        Given: Trivial module that user has no intent of running forward with.
        """

        class Trivial(torch.nn.Module):
            # NOTE: This module will ALWAYS fail when forward is called,
            #       but it's ok since the user has no intent to do so
            pass

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._trivial = Trivial()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        """
        When: Create quantsim
        Then: Quantsim shouldn't complain about not defining quantized definition for trivial modules
        """
        sim = QuantizationSimModel(Model(), torch.randn(10, 10))
        assert isinstance(sim.model._trivial, Trivial)
        assert isinstance(sim.model.linear, QuantizedLinear)
        assert isinstance(
            sim.model.linear.param_quantizers["weight"], QuantizeDequantize
        )
        assert isinstance(sim.model.linear.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.linear.output_quantizers[0], QuantizeDequantize)

    def test_already_quantized_model(self):
        """
        Given: The model already consists of quantized modules
        When: Create quantsim with the model
        Then: Throw runtime error
        """
        model = torch.nn.Sequential(
            QuantizedConv2d(3, 3, 3),
            torch.nn.ReLU(),
        )
        dummy_input = torch.randn(1, 3, 224, 224)

        with pytest.raises(RuntimeError):
            _ = QuantizationSimModel(model, dummy_input)

        """
        Given: The model already consists of quantizers
        When: Create quantsim with the model
        Then: Throw runtime error
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            QuantizeDequantize((), 0, 255, False),
        )

        with pytest.raises(RuntimeError):
            _ = QuantizationSimModel(model, dummy_input)

        """
        Given: The model itself is a quantized module
        When: Create quantsim with the model
        Then: Throw runtime error
        """
        model = QuantizedConv2d(3, 3, 3)

        with pytest.raises(RuntimeError):
            _ = QuantizationSimModel(model, dummy_input)

        """
        Given: The model itself is a quantizer
        When: Create quantsim with the model
        Then: Throw runtime error
        """
        model = QuantizeDequantize((), 0, 255, False)

        with pytest.raises(RuntimeError):
            _ = QuantizationSimModel(model, dummy_input)

    def test_quantize_constant_python_float(self):
        """Test that model input quantizers are enabled correctly when using different constant types"""
        dummy_input = torch.randn(2, 1)

        """
        Given: A model with python float constant
        When: Instantiate quantsim and run compute_encodings
        Then: 1. The input quantizer quantizing buffer constant should be enabled
              2. The quantizer should not be initialized
        """

        class PythonFloatModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.module = custom.Add()

            def forward(self, *inputs):
                x = self.module(inputs[0], 2.0)
                return x

        model = PythonFloatModel()
        sim = QuantizationSimModel(
            model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=dummy_input,
            in_place=True,
        )
        sim.compute_encodings(lambda m, d: m(d), dummy_input)
        sim.model(dummy_input)

        assert sim.model.module.input_quantizers[1] is not None
        assert not sim.model.module.input_quantizers[1].is_initialized()

    def test_compute_encodings_optional_arg(self):
        """
        Given: Two quantsims created with identical model & config
        """
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim_a = QuantizationSimModel(model, dummy_input)
        sim_b = QuantizationSimModel(model, dummy_input)

        """
        When: Run compute_encodings with second argument omitted in one quantsim and not in the other
        Then: The quantizers in both quantsims should have the same encodings
        """
        sim_a.compute_encodings(lambda model: model(dummy_input))
        sim_b.compute_encodings(
            lambda model, x: model(x), forward_pass_callback_args=dummy_input
        )

        for qtzr_a, qtzr_b in zip(sim_a.model.modules(), sim_b.model.modules()):
            if isinstance(qtzr_a, AffineQuantizerBase):
                assert torch.equal(qtzr_a.get_scale(), qtzr_b.get_scale())
                assert torch.equal(qtzr_a.get_offset(), qtzr_b.get_offset())
                assert torch.equal(qtzr_a.get_min(), qtzr_b.get_min())
                assert torch.equal(qtzr_a.get_max(), qtzr_b.get_max())

    @pytest.mark.parametrize(
        "data_type", [QuantizationDataType.int, QuantizationDataType.float]
    )
    def test_fold_param_quantizers(self, tmpdir, data_type):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
        )
        x = torch.randn(10, 10)
        sim = QuantizationSimModel(
            model,
            x,
            default_param_bw=16,
            default_output_bw=16,
            default_data_type=data_type,
        )
        sim.compute_encodings(lambda model: model(x))

        sim.export(tmpdir, "before_fold", x)

        """
        When: Call fold_param_quantizers()
        Then: 1. All param quantizers should be folded to the parameter
              2. Export artifact of sim.export() should not be affected
        """
        sim.fold_param_quantizers()
        assert sim.model[0].param_quantizers["weight"] is None
        assert isinstance(sim.model[0].weight, DequantizedTensor)

        sim.export(tmpdir, "after_fold", x)

        with open(os.path.join(tmpdir, "before_fold.encodings")) as f:
            encodings_before_fold = json.load(f)
        with open(os.path.join(tmpdir, "after_fold.encodings")) as f:
            encodings_after_fold = json.load(f)

        assert encodings_before_fold == encodings_after_fold

        # trivial sanity check
        assert [enc["name"] for enc in encodings_before_fold["param_encodings"]] == [
            "0.weight"
        ]

    @pytest.mark.parametrize(
        "module_factory, input_factory",
        [
            (lambda: nn.Upsample(scale_factor=2), lambda: randn(1, 1, 10, 10)),
            (
                lambda: nn.UpsamplingBilinear2d(scale_factor=2),
                lambda: randn(1, 1, 10, 10),
            ),
            (
                lambda: nn.UpsamplingNearest2d(scale_factor=2),
                lambda: randn(1, 1, 10, 10),
            ),
            (lambda: nn.ReLU(), lambda: randn(1, 1, 10, 10)),
            # (lambda torchvision.transforms.Resize(),        lambda: ...),
            # TODO: Need to enable output quantization of interpolation layers
            #       in htp config file to pass below test cases
            # (lambda: nn.MaxPool1d(3), lambda: randn(1, 3, 100)),
            # (lambda: nn.MaxPool2d(3), lambda: randn(1, 3, 10, 10)),
            # (lambda: nn.MaxPool3d(3), lambda: randn(1, 3, 5, 5, 5)),
            # (lambda: nn.AvgPool2d(3), lambda: randn(1, 3, 5, 5, 5)),
            # (lambda: nn.AvgPool2d(3), lambda: randn(1, 3, 5, 5, 5)),
        ],
    )
    def test_htp_interpolation_tie_encodings(self, module_factory, input_factory):
        """
        Purpose: HTP treats onnx::Resize as data movement op, and therefore
                 requires its input/output to share the same encoding
        """

        """
        Given: Modules that get lowered into onnx::Resize
        """
        model = torch.nn.Sequential(module_factory())
        inputs = input_factory()

        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)

        """
        When: Create quantsim
        Then: Input/output should share the same quantizer
        """
        sim = QuantizationSimModel(model, inputs, config_file="htp_v81")
        assert sim.model[0].input_quantizers[0] is sim.model[0].output_quantizers[0]

        """
        When: Export
        Then: Input/output should share the same encodings
        """
        sim.compute_encodings(lambda model: model(*inputs))

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(tmpdir, "resize", inputs)

            with open(os.path.join(tmpdir, "resize.encodings")) as f:
                encodings = json.load(f)

        inp_enc, out_enc = encodings["activation_encodings"]
        inp_enc.pop("name")
        out_enc.pop("name")
        assert inp_enc == out_enc

    def test_conv_relu_supergroup(self, tmp_path: pathlib.Path):
        """
        When: Create quantsim with HTP V69 config or lower
        Then:
          - Conv-Relu should NOT be a supergroup
          - Conv output quantizer must be tied with Relu output quantizer
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.ReLU(),
        )
        x = torch.randn(1, 3, 10, 10)
        sim = QuantizationSimModel(model, x, config_file="htp_v69")
        (conv_output_qtzr,) = sim.model[0].output_quantizers
        (relu_output_qtzr,) = sim.model[1].output_quantizers
        assert conv_output_qtzr is relu_output_qtzr

        """
        When: Compute_encodings and run QAT
        Then: Conv output encoding should remain non-negative
        """
        sim.compute_encodings(lambda model: model(x))
        old_params = {
            name: param.clone().detach() for name, param in sim.model.named_parameters()
        }
        optim = torch.optim.AdamW(sim.model.parameters())

        for _ in range(5):
            x = torch.randn(1, 3, 10, 10)
            sim_out = sim.model(x)

            with torch.no_grad():
                fp_out = model(x)

            loss = torch.nn.functional.mse_loss(sim_out, fp_out)
            loss.backward()
            optim.step()
            optim.zero_grad()

        assert conv_output_qtzr.min == relu_output_qtzr.min == 0
        # Sanity check to prevent trivial pass
        assert not all(
            torch.equal(param_old, param_new)
            for param_old, param_new in zip(old_params.values(), sim.model.parameters())
        )

        """
        When: Export
        Then: Conv and Relu output encoding should remain identical
        """
        sim.export(tmp_path, "export", x)

        with open(tmp_path / "export.encodings") as f:
            encodings = json.load(f)

        _, conv_out_enc, relu_out_enc = encodings["activation_encodings"]
        conv_out_enc.pop("name")
        relu_out_enc.pop("name")
        assert conv_out_enc == relu_out_enc

        """
        When: Create quantsim with HTP V73 config or higher
        Then:
          - Conv-Relu should be a supergroup
          - Conv input quantizer must NOT be tied with Relu output quantizer
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.ReLU(),
        )
        x = torch.randn(1, 3, 10, 10)
        sim = QuantizationSimModel(model, x, config_file="htp_v73")
        (conv_input_qtzr,) = sim.model[0].input_quantizers
        (conv_output_qtzr,) = sim.model[0].output_quantizers
        (relu_output_qtzr,) = sim.model[1].output_quantizers
        assert conv_output_qtzr is None
        assert conv_input_qtzr is not relu_output_qtzr

        """
        Given: model as below

          ... -> conv -> q_out1 --+--> relu ----> q_out2 -> [output_1]
                                  +--> softmax -> q_out3 -> [output_2]

          where q_out2 has fixed encoding constraints [0, ?]
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()
                self.softmax = torch.nn.Softmax()

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x), self.softmax(x)

        """
        When: Create quantsim with HTP V69 config
        Then: q_out1 should not be tied with q_out2
        """
        model = Model()
        x = torch.randn(1, 3, 10, 10)
        sim = QuantizationSimModel(model, x, config_file="htp_v69")
        (conv_output_qtzr,) = sim.model.conv.output_quantizers
        (relu_output_qtzr,) = sim.model.relu.output_quantizers
        assert conv_output_qtzr is not relu_output_qtzr


class TestQuantsimUtilities:
    def test_populate_marker_map(self):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(model, dummy_input)
        conv_layer = sim.model.conv
        for name, module in sim.model.named_modules():
            if module is conv_layer:
                conv_name = name
                break
        assert conv_name not in sim._module_marker_map.keys()
        sim.run_modules_for_traced_custom_marker([conv_layer], dummy_input)
        assert conv_name in sim._module_marker_map.keys()
        assert torch.equal(
            sim._module_marker_map[conv_name](dummy_input),
            conv_layer.get_original_module()(dummy_input),
        )

    def test_get_leaf_module_to_name_map(self):
        model = test_models.NestedConditional()
        dummy_input = torch.rand(1, 3), torch.tensor([True])
        sim = QuantizationSimModel(model, dummy_input)
        leaf_modules = sim._get_leaf_module_to_name_map()
        for name, module in sim.model.named_modules():
            if isinstance(module, BaseQuantizationMixin):
                assert module in leaf_modules.keys()
                assert leaf_modules[module] == name

    @pytest.mark.skip
    def test_supergroup_bfs(self):
        """
        Given: model as below
            [input] -+--> conv1 --> relu1 ---> sum --> (output)
                     +--> conv2 --> relu2 ------^

        When: Call modules in a BFS-order: 1) conv1 2) conv2 3) relu1 4) relu4
        Then: Output quantizers of conv1 and conv2 shouldn't be instantiated

        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x1 = self.relu1(x1)
                x2 = self.relu2(x2)
                return x1 + x2

        model = Model()
        x = torch.randn(1, 3, 24, 24)
        sim = QuantizationSimModel(model, x)

        assert sim.model.conv1.output_quantizers[0] is None
        assert sim.model.conv2.output_quantizers[0] is None


class TestEncodingPropagation:
    def test_output(self):
        """
        Given: model as below

                   +-> q_in1 -> conv1 -> relu1 ---> q_out1 -------v
          [input] -+                                           concat -> q_out3 -> [output]
                   +-> q_in2 -> conv2 -> relu2 ---> q_out2 -------^
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()
                self.cat = custom.Concat()

            def forward(self, x):
                x1 = x2 = x
                x1 = self.conv1(x1)
                x2 = self.conv2(x2)
                x1 = self.relu1(x1)
                x2 = self.relu2(x2)
                return self.cat(x1, x2)

        model = Model()
        x = torch.randn(1, 3, 24, 24)
        sim = QuantizationSimModel(model, x)

        """
        When: Call propagate_output_encodings(concat)

        Then: q_out1 and q_out2 are replaced with q_out3 as below

                   +-> q_in1 -> conv1 -> relu1 -> **q_out3** -----v
          [input] -+                                           concat -> q_out3- > [output]
                   +-> q_in2 -> conv2 -> relu2 -> **q_out3** -----^
        """

        orig_q_in1 = sim.model.conv1.input_quantizers[0]
        orig_q_in2 = sim.model.conv2.input_quantizers[0]
        orig_q_out3 = sim.model.cat.output_quantizers[0]

        propagate_output_encodings(sim, custom.Concat)

        q_in1 = sim.model.conv1.input_quantizers[0]
        q_in2 = sim.model.conv2.input_quantizers[0]
        q_out1 = sim.model.relu1.output_quantizers[0]
        q_out2 = sim.model.relu2.output_quantizers[0]
        q_out3 = sim.model.cat.output_quantizers[0]

        # q_out1 == q_out2 == q_out3
        assert q_out1 is q_out3
        assert q_out2 is q_out3

        # q_in1, q_in2, and q_out3 stay unchanged
        assert q_in1 is orig_q_in1
        assert q_in2 is orig_q_in2
        assert q_out3 is orig_q_out3

    @pytest.mark.parametrize("permute_impl", [custom.Permute(), torch.permute])
    def test_math_invariant(self, permute_impl):
        """
        Given: model as below

                   +-> q_in1 -> conv1 ---> relu1 -> q_out1 ------v
          [input] -+                                          concat -> q_out3 -> [output]
                   +-> q_in2 -> reshape -> permute --------------^
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.relu1 = torch.nn.ReLU()

                self.reshape = custom.Reshape()
                self.permute = permute_impl

                self.cat = custom.Concat()

            def forward(self, x):
                # assert x.shape[1:] == torch.Size([3, 24, 24])
                x1 = x2 = x
                x1 = self.conv1(x1)
                x1 = self.relu1(x1)

                x2 = self.reshape(x2, (-1, 24, 24, 3))
                x2 = self.permute(x2, (0, 3, 1, 2))
                return self.cat(x1, x2)

        model = Model()
        x = torch.randn(1, 3, 24, 24)
        sim = QuantizationSimModel(model, x)

        """
        When: Call propagate_output_encodings(concat)

        Then: q_out1 and q_in2 are replaced with q_out3 as below

                   +-> q_in1 -> conv1 ---> relu1 -----> **q_out3**- --------v
          [input] -+                                                     concat -> q_out3 -> [output]
                   +-> **q_out3** -> reshape -> transpose -> permute -------^
        """
        orig_q_in1 = sim.model.conv1.input_quantizers[0]
        orig_q_out3 = sim.model.cat.output_quantizers[0]

        propagate_output_encodings(sim, custom.Concat)

        q_in1 = sim.model.conv1.input_quantizers[0]
        q_in2 = sim.model.reshape.input_quantizers[0]
        q_out1 = sim.model.relu1.output_quantizers[0]
        q_out3 = sim.model.cat.output_quantizers[0]

        # q_out1 == q_in2 == q_out3
        assert q_out1 is q_out3
        assert q_in2 is q_out3

        # q_in1 and q_out3 stay unchanged
        assert q_in1 is orig_q_in1
        assert q_out3 is orig_q_out3

    def test_concat_tree(self):
        """
        Given: model as below

                    +-> q_in1a -> conv1a -> q_out1a -> concat1 -> q_out1c -> reshape --+
                    +-> q_in1b -> conv1b -> q_out1b ------^                            v
          [input] --+                                                               concat3 -> q_out3 -> [output]
                    +-> q_in2a -> conv2a -> q_out2a -> concat2 -> q_out2c -------------^
                    +-> q_in2b -> conv2b -> q_out2b ------^
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1a = torch.nn.Conv2d(3, 3, 3)
                self.conv1b = torch.nn.Conv2d(3, 3, 3)
                self.conv2a = torch.nn.Conv2d(3, 3, 3)
                self.conv2b = torch.nn.Conv2d(3, 3, 3)

                self.reshape = custom.Reshape()
                self.permute = custom.Permute()

                self.cat1 = custom.Concat()
                self.cat2 = custom.Concat()
                self.cat3 = custom.Concat()

            def forward(self, x):
                # assert x.shape[1:] == torch.Size([3, 24, 24])
                x1a = x1b = x2a = x2b = x

                x1a = self.conv1a(x1a)
                x1b = self.conv1b(x1b)
                x1 = self.cat1(x1a, x1b)
                x1 = self.reshape(x1, (-1, 22, 22, 3))
                x1 = self.permute(x1, (0, 3, 1, 2))

                x2a = self.conv2a(x2a)
                x2b = self.conv2b(x2b)
                x2 = self.cat2(x2a, x2b)

                return self.cat3(x1, x2)

        model = Model()
        x = torch.randn(1, 3, 24, 24)
        sim = QuantizationSimModel(model, x)
        sim.model.reshape.output_quantizers[0] = None
        sim.model.permute.output_quantizers[0] = None

        """
        When: Call propagate_output_encodings(concat)

        Then: All q_out{*} are replaced with q_out3 as below

                    +-> q_in1a -> conv1a -> *q_out3* -> concat1 -> *q_out3* -> reshape --+
                    +-> q_in1b -> conv1b -> *q_out3* ------^                             v
          [input] --+                                                                 concat3 -> q_out3 -> [output]
                    +-> q_in2a -> conv2a -> *q_out3* -> concat2 -> *q_out3* -------------^
                    +-> q_in2b -> conv2b -> *q_out3* ------^
        """
        orig_q_out3 = sim.model.cat3.output_quantizers[0]

        propagate_output_encodings(sim, custom.Concat)

        q_out1a = sim.model.conv1a.output_quantizers[0]
        q_out1b = sim.model.conv1b.output_quantizers[0]
        q_out2a = sim.model.conv2a.output_quantizers[0]
        q_out2b = sim.model.conv2b.output_quantizers[0]
        q_out1 = sim.model.cat1.output_quantizers[0]
        q_out2 = sim.model.cat2.output_quantizers[0]
        q_out3 = sim.model.cat3.output_quantizers[0]

        assert q_out1a is q_out3
        assert q_out1b is q_out3
        assert q_out2a is q_out3
        assert q_out2b is q_out3
        assert q_out1 is q_out3
        assert q_out2 is q_out3

        # q_out3 stay unchanged
        assert q_out3 is orig_q_out3

    def test_variadic_qmodules(self):
        """
        Given: model as below

           [x] -+                                                                   +---------------> [output1]
           [y] -+-> q_in -> concat1 -> q_out1 -> conv -> q_out2 -> split -> q_out3 -+-+
           [z] -+                                                                   +-+-> concat2 -> q_out4 -> [output2]
        """

        # NOTE: Input-variadic qmodule Concat and output-variadic qmodule Split
        #       has only one input/output quantizer that covers variable number of input/output tensors.
        #       This test checks if propagate_output_encodings can properly handle these variadic operators

        # FIXME: Currently, propagate_output_encodings doesn't work with models with torch.split
        #        because connected graph fails to create a computation graph of torch.split correctly.

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cat1 = custom.Concat()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                # TODO
                # self.split = custom.Split()
                # self.cat2 = custom.Concat()

            def forward(self, *tensors):
                t = self.cat1(*tensors)
                t = self.conv(t)
                return t
                # TODO
                # x, y, z = self.split(t, 1)
                # return self.cat2(x, y, z)

        model = Model()
        x = torch.randn(1, 3, 24, 24)
        y = torch.randn(1, 3, 24, 24)
        z = torch.randn(1, 3, 24, 24)
        sim = QuantizationSimModel(model, (x, y, z))

        """
        When: Call propagate_output_encodings
        Then:

           [x] -+                                                                         +---------------> [output1]
           [y] -+-> *q_out1* -> concat1 -> q_out1 -> conv -> q_out2 -> split -> *q_out4* -+-+
           [z] -+                                                                         +-+-> concat2 -> q_out4 -> [output2]
        """
        propagate_output_encodings(sim, custom.Concat)
        assert sim.model.cat1.input_quantizers[0] is sim.model.cat1.output_quantizers[0]
        # assert sim.model.split.output_quantizers[0] is sim.model.cat2.output_quantizers[0] TODO

        ########################################################################

        """
        Given: model as below

           [x] ---> q_in1 -> conv -> q_out1 --+
           [y] -+-> q_in2 --------------------+-> concat -> q_out2 -> [output]
           [z] -+
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.cat = custom.Concat()

            def forward(self, x, y, z):
                x = self.conv(x)
                return self.cat(x, y, z)

        model = Model()
        x = torch.randn(1, 3, 26, 26)
        y = torch.randn(1, 3, 24, 24)
        z = torch.randn(1, 3, 24, 24)
        sim = QuantizationSimModel(model, (x, y, z))

        """
        When: Call propagate_output_encodings
        Then:

           [x] ---> q_in1 -> conv -> *q_out2* --+
           [y] -+-> *q_out2* -------------------+-> concat -> q_out2 -> [output]
           [z] -+
        """
        propagate_output_encodings(sim, custom.Concat)
        assert sim.model.conv.output_quantizers[0] is sim.model.cat.output_quantizers[0]
        assert sim.model.cat.input_quantizers[0] is sim.model.cat.output_quantizers[0]

    def test_functional(self):
        """
        Given: Model as below, where reshape and permute are functional operators.
               Note that there is no parent nn.Module for the second input of concat
               to propagate the output encoidngs to.

          [input] -> reshape -> permute -> concat -> q_out -> [output]
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cat = custom.Concat()

            def forward(self, x):
                x1 = x2 = x
                x2 = torch.reshape(x2, (-1, 24, 24, 3))
                x2 = torch.permute(x2, (0, 3, 1, 2))
                return self.cat(x1, x2)

        model = Model()
        x = torch.randn(1, 3, 24, 24)
        sim = QuantizationSimModel(model, x)

        """
        When: Call propagate_output_encodings(concat)
        Then: Shouldn't throw runtime error, even though there is no ancestor
              to propagate the output encodings to.
        """
        propagate_output_encodings(sim, custom.Concat)

    def test_skip_torch_encodings(self):
        @contextlib.contextmanager
        def swap_skip_torch_encodings(skip_torch_encodings):
            from aimet_torch._base import quantsim

            old_setting = quantsim.SKIP_TORCH_ENCODINGS_EXPORT
            quantsim.SKIP_TORCH_ENCODINGS_EXPORT = skip_torch_encodings

            yield

            quantsim.SKIP_TORCH_ENCODINGS_EXPORT = old_setting

        model = test_models.SingleResidualWithAvgPool()
        dummy_input = torch.randn(1, 3, 28, 28)

        qsim = QuantizationSimModel(model, dummy_input)
        qsim.compute_encodings(lambda m, _: m(dummy_input), None)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            swap_skip_torch_encodings(False),
        ):
            qsim.export(temp_dir, "model_export", dummy_input)
            assert os.path.isfile(
                os.path.join(temp_dir, "model_export_torch.encodings")
            )

        with tempfile.TemporaryDirectory() as temp_dir, swap_skip_torch_encodings(True):
            qsim.export(temp_dir, "model_export", dummy_input)
            assert not os.path.isfile(
                os.path.join(temp_dir, "model_export_torch.encodings")
            )

    def test_torch_encodings_parity(self):
        @contextlib.contextmanager
        def swap_encoding_version(encoding_version):
            from aimet_common import quantsim as aimet_common_quantsim

            old_setting = aimet_common_quantsim.encoding_version
            aimet_common_quantsim.encoding_version = encoding_version

            yield

            aimet_common_quantsim.encoding_version = old_setting

        model = test_models.SingleResidualWithAvgPool()
        dummy_input = torch.randn(1, 3, 28, 28)

        qsim = QuantizationSimModel(model, dummy_input)
        qsim.compute_encodings(lambda m, _: m(dummy_input), None)

        with tempfile.TemporaryDirectory() as temp_dir, swap_encoding_version(False):
            with swap_encoding_version("0.6.1"):
                qsim.export(temp_dir, "model_export_0_6_1", dummy_input)
            with swap_encoding_version("1.0.0"):
                qsim.export(temp_dir, "model_export_1_0_0", dummy_input)

            with open(
                os.path.join(temp_dir, "model_export_0_6_1_torch.encodings")
            ) as encodings_0_6_1_file:
                encodings_0_6_1 = json.load(encodings_0_6_1_file)
            with open(
                os.path.join(temp_dir, "model_export_1_0_0_torch.encodings")
            ) as encodings_1_0_0_file:
                encodings_1_0_0 = json.load(encodings_1_0_0_file)

            assert (
                encodings_0_6_1["activation_encodings"]
                == encodings_1_0_0["activation_encodings"]
            )
            assert (
                encodings_0_6_1["param_encodings"] == encodings_1_0_0["param_encodings"]
            )

    def test_shared_module(self):
        """
        Given: Model with ambiguous child module ownership.

                        Model
                          |
                     +----+----+
                     |         V
                     |     ModuleList
                     V         |
                 Sequential <--+
                     |
                  +--+--+
                  V     V
               Linear  ReLU

        (Note that Sequential is a child of Model and ModuleList at the same time)
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                )
                self.module_list = torch.nn.ModuleList([self.seq])

        """
        When: The shared child modules are NOT reused during forward
        Then: Quantsim should be instantiated normally
        """

        class _Model(Model):
            def forward(self, x):
                return self.seq(x)

        sim = QuantizationSimModel(_Model(), torch.randn(10, 10))

        assert sim.model.seq is sim.model.module_list[0]

        assert isinstance(sim.model.seq[0], QuantizedLinear)
        assert isinstance(
            sim.model.seq[0].param_quantizers["weight"], QuantizeDequantize
        )
        assert isinstance(sim.model.seq[0].input_quantizers[0], QuantizeDequantize)
        assert sim.model.seq[0].output_quantizers[0] is None

        assert isinstance(sim.model.seq[1], QuantizedReLU)
        assert sim.model.seq[1].input_quantizers[0] is None
        assert isinstance(sim.model.seq[1].output_quantizers[0], QuantizeDequantize)

    def test_nested_input(self):
        class MyLinear(torch.nn.Module):
            def forward(self, xy: tuple[torch.Tensor, torch.Tensor], z: torch.Tensor):
                x, y = xy
                return torch.nn.functional.linear(x, y, z)

        @QuantizationMixin.implements(MyLinear)
        class QuantizedMyLinear(QuantizationMixin, MyLinear):
            def __quant_init__(self):
                super().__quant_init__()

                # Declare the number of input/output quantizers
                self.input_quantizers = torch.nn.ModuleList([None, None, None])
                self.output_quantizers = torch.nn.ModuleList([None])

            def forward(self, xy: tuple[torch.Tensor, torch.Tensor], z: torch.Tensor):
                x, y = xy

                if self.input_quantizers[0]:
                    x = self.input_quantizers[0](x)

                if self.input_quantizers[1]:
                    y = self.input_quantizers[1](y)

                if self.input_quantizers[2]:
                    z = self.input_quantizers[2](z)

                out = super().forward(xy, z)

                if self.output_quantizers[0]:
                    out = self.output_quantizers[0](out)

                return out

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = MyLinear()

            def forward(self, *args):
                return self.linear(*args)

        """
        When: Leaf module takes nested tuple of tensors as input
        Then: Quantsim shouldn't fail
        """
        model = Model()
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        z = torch.randn(10, 10)
        nested_input = ((x, y), z)
        sim = QuantizationSimModel(model, nested_input)

        assert isinstance(sim.model.linear.input_quantizers[0], QuantizeDequantize)
        assert isinstance(sim.model.linear.input_quantizers[1], QuantizeDequantize)
        assert isinstance(sim.model.linear.input_quantizers[2], QuantizeDequantize)
        assert isinstance(sim.model.linear.output_quantizers[0], QuantizeDequantize)

    def test_export_with_zero_point_shift(self):
        torch.manual_seed(0)

        dummy_input = torch.randn(1, 3, 4, 4)
        model = ConvModel()
        qsim = QuantizationSimModel(
            model, dummy_input, config_file=get_path_for_per_channel_config()
        )
        qsim.model.conv.param_quantizers["weight"] = QuantizeDequantize(
            shape=qsim.model.conv.param_quantizers["weight"].shape,
            bitwidth=2,
            symmetric=True,
            zero_point_shift=0.5,
        )
        qsim.compute_encodings(lambda m: m(dummy_input))
        out_1 = qsim.model(dummy_input)
        with tempfile.TemporaryDirectory() as tmpdir:
            qsim.export(tmpdir, "zero_point_shift_0.5_export", dummy_input)
            qsim.save_encodings_to_json(tmpdir, "zero_point_shift_0.5_encodings")
            with open(
                os.path.join(tmpdir, "zero_point_shift_0.5_export.encodings"), "r"
            ) as f:
                encodings = json.load(f)
                assert encodings["param_encodings"][0]["offset"][0] == -2
                assert encodings["param_encodings"][0]["zero_point_shift"][0] == 0.5

            with open(
                os.path.join(tmpdir, "zero_point_shift_0.5_encodings.json"), "r"
            ) as f:
                encodings = json.load(f)
                assert encodings["param_encodings"]["conv.weight"][0]["offset"] == -2

            qsim_2 = QuantizationSimModel(
                model, dummy_input, config_file=get_path_for_per_channel_config()
            )
            qsim_2.model.conv.param_quantizers["weight"] = QuantizeDequantize(
                shape=qsim.model.conv.param_quantizers["weight"].shape,
                bitwidth=2,
                symmetric=True,
                zero_point_shift=0.5,
            )

            assert not qsim_2.model.conv.param_quantizers["weight"].get_encodings()
            qsim_2.load_encodings(
                os.path.join(tmpdir, "zero_point_shift_0.5_encodings.json")
            )
            qsim_weight_encoding = qsim.model.conv.param_quantizers[
                "weight"
            ].get_encodings()
            qsim_2_weight_encoding = qsim_2.model.conv.param_quantizers[
                "weight"
            ].get_encodings()
            assert qsim_weight_encoding.min == qsim_2_weight_encoding.min
            assert qsim_weight_encoding.max == qsim_2_weight_encoding.max
            assert qsim_weight_encoding.scale == qsim_2_weight_encoding.scale
            assert qsim_weight_encoding.offset == qsim_2_weight_encoding.offset

            out_2 = qsim_2.model(dummy_input)
            assert torch.allclose(out_1, out_2, atol=1e-7)


class ReshapeConv(torch.nn.Module):
    def __init__(self, functional: bool):
        super().__init__()
        if functional:
            self.reshape = torch.reshape
        else:
            self.reshape = custom.Reshape()
        self.conv = torch.nn.Conv1d(3, 3, 3)

    def forward(self, x):
        x = self.reshape(x, (1, 3, -1))
        return self.conv(x)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: ReshapeConv(functional=True),
        lambda: ReshapeConv(functional=False),
    ],
)
def test_input_quantizer_enabling(model_factory):
    """
    Given: Model whose input is fed into functional data movement op
    """
    model = model_factory()
    x = torch.randn(3, 100)

    """
    When: Create quantsim
    Then: There should be exactly one input quantizer
    """
    sim = QuantizationSimModel(model, x)
    input_qtzrs = itertools.chain(
        *(
            [qtzr for qtzr in qmodule.input_quantizers if qtzr]
            for qmodule in sim.qmodules()
        )
    )
    assert len(list(input_qtzrs)) == 1

    """
    When: Export to onnx QDQ
    Then: All inputs/outputs should be associated with QDQ
    """
    sim.compute_encodings(lambda model: model(x))

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "model.onnx")
        aimet_torch.onnx.export(sim.model, x, onnx_path)
        onnx_model = onnx.load_model(onnx_path)

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    dtypes = {
        val.name: val.type.tensor_type.elem_type for val in onnx_model.graph.value_info
    }

    for node in onnx_model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue

        if node.input and dtypes[node.input[0]] == onnx.TensorProto.FLOAT:
            producer = next(
                dq for dq in onnx_model.graph.node if dq.output[:1] == node.input[:1]
            )
            assert producer.op_type == "DequantizeLinear"

        if node.output and dtypes[node.output[0]] == onnx.TensorProto.FLOAT:
            consumer = next(
                q for q in onnx_model.graph.node if node.output[:1] == q.input[:1]
            )
            assert consumer.op_type == "QuantizeLinear"
