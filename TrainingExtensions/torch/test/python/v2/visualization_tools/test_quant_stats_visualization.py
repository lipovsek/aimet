# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import re
import tempfile
import os.path
import torch
from ...models.test_models import TinyModel
from ...models.test_models import ModelWithUnusedMatmul
from aimet_torch.common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.visualization_tools import visualize_stats, visualize_advanced_stats
from aimet_torch.common.quantsim_config.utils import get_path_for_per_channel_config


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model performance given dummy input
    :param model: PyTorch model
    :param dummy_input: dummy input to model.
    """
    model.eval()
    for _ in range(2):
        with torch.no_grad():
            model(dummy_input)
    return 0.8


def evaluate_2(model: torch.nn.Module, dummy_input: tuple):
    """
    Helper function to evaluate model performance given dummy inputs as a tuple
    :param model: PyTorch model
    :param dummy_input: dummy inputs to model as a tuple.
    """
    model.eval()
    for _ in range(2):
        with torch.no_grad():
            model(*dummy_input)
    return 0.8


class TestQuantStatsVisualization:
    @pytest.mark.parametrize(
        "quant_scheme",
        [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced],
    )
    @pytest.mark.parametrize("config_file", [None, get_path_for_per_channel_config()])
    def test_visualize_stats(self, quant_scheme, config_file):
        """test visualize_stats API"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy_input,
            quant_scheme=quant_scheme,
            config_file=config_file,
        )
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visualize_stats(
                sim,
                dummy_input,
                save_path=os.path.join(tmp_dir, "test_visualize_stats.html"),
            )
            assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))

    @pytest.mark.parametrize(
        "quant_scheme",
        [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced],
    )
    @pytest.mark.parametrize("config_file", [None, get_path_for_per_channel_config()])
    def test_visualize_advanced_stats(self, quant_scheme, config_file):
        """test visualize_advanced_stats API"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy_input,
            quant_scheme=quant_scheme,
            config_file=config_file,
        )
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visualize_advanced_stats(
                sim,
                dummy_input,
                save_path=os.path.join(tmp_dir, "test_visualize_stats.html"),
            )
            assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))

    @pytest.mark.parametrize(
        "quant_scheme",
        [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced],
    )
    @pytest.mark.parametrize("function", [visualize_stats, visualize_advanced_stats])
    def test_not_calibrated_error(self, quant_scheme, function):
        """Check whether an exception is raised if QuantSim is not calibrated"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(
            model, dummy_input=dummy_input, quant_scheme=quant_scheme
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(RuntimeError):
                function(
                    sim,
                    dummy_input,
                    save_path=os.path.join(tmp_dir, "test_visualize_stats.html"),
                )

    @pytest.mark.parametrize("function", [visualize_stats, visualize_advanced_stats])
    def test_not_quantsim_object_error(self, function):
        """Check whether the input is a QuantizationSimModel instance"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(TypeError):
                function(
                    model,
                    dummy_input,
                    save_path=os.path.join(tmp_dir, "test_visualize_stats.html"),
                )

    @pytest.mark.parametrize("function", [visualize_stats, visualize_advanced_stats])
    def test_not_a_directory_error(self, function):
        """Raise exception if directory corresponding to save_path does not exist"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(
            model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf
        )
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = tmp_dir
        # re.escape: on Windows the temp path contains backslashes (e.g. \\Users),
        # which are invalid regex escapes for pytest.raises(match=...).
        with pytest.raises(
            NotADirectoryError, match=re.escape(f"'{tmp}' is not a directory.")
        ):
            function(
                sim,
                dummy_input,
                save_path=os.path.join(tmp, "test_visualize_stats.html"),
            )

    @pytest.mark.parametrize("function", [visualize_stats, visualize_advanced_stats])
    def test_no_html_extension_error(self, function):
        """Raise exception if provided path does not end with .html"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(
            model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf
        )
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="'save_path' must end with '.html'."):
                function(
                    sim,
                    dummy_input,
                    save_path=os.path.join(tmp_dir, "test_visualize_stats.jpg"),
                )

    @pytest.mark.parametrize(
        "quant_scheme",
        [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced],
    )
    @pytest.mark.parametrize("function", [visualize_stats, visualize_advanced_stats])
    def test_model_with_unused_matmul(self, quant_scheme, function):
        """Check that the visualization is generated even when there exists an unused matmul"""
        model = ModelWithUnusedMatmul()
        dummy_input = (torch.randn(10), torch.randn(10))
        sim = QuantizationSimModel(
            model, dummy_input=dummy_input, quant_scheme=quant_scheme
        )
        sim.compute_encodings(evaluate_2, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            function(
                sim,
                dummy_input,
                save_path=os.path.join(tmp_dir, "test_visualize_stats.html"),
            )
            assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))
