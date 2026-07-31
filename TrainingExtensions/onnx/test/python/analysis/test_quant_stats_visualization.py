# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import copy
import os.path
import re
import tempfile

import numpy as np
import pytest
import torch

from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
from aimet_onnx.analysis import visualize_stats

from ..models import models_for_tests


def _tiny_sim():
    """Build and calibrate a small ONNX QuantizationSimModel for testing."""
    dummy_input = torch.randn(1, 3, 32, 32)
    model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
    dummy_input_dict = {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}
    sim = QuantizationSimModel(copy.deepcopy(model))
    with compute_encodings(sim):
        sim.session.run(None, dummy_input_dict)
    return sim


class TestQuantStatsVisualization:
    def test_visualize_stats(self):
        """Both activation and weight HTML files are produced from a calibrated sim."""
        sim = _tiny_sim()
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "quant_stats_visualization.html")
            visualize_stats(sim, save_path=save_path)
            assert os.path.isfile(
                os.path.join(tmp_dir, "quant_stats_visualization_activations.html")
            )
            assert os.path.isfile(
                os.path.join(tmp_dir, "quant_stats_visualization_weights.html")
            )

    def test_not_calibrated_error(self):
        """A RuntimeError is raised if the sim has not been calibrated."""
        dummy_input = torch.randn(1, 3, 32, 32)
        model = models_for_tests._convert_to_onnx(
            models_for_tests.TinyModel(), dummy_input
        )
        sim = QuantizationSimModel(copy.deepcopy(model))
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(RuntimeError):
                visualize_stats(
                    sim, save_path=os.path.join(tmp_dir, "quant_stats.html")
                )

    def test_not_quantsim_object_error(self):
        """A TypeError is raised if the input is not a QuantizationSimModel."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(TypeError):
                visualize_stats(
                    object(), save_path=os.path.join(tmp_dir, "quant_stats.html")
                )

    def test_not_a_directory_error(self):
        """A NotADirectoryError is raised if save_path's directory does not exist."""
        sim = _tiny_sim()
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = tmp_dir
        # re.escape: on Windows the temp path contains backslashes (e.g. \\Users),
        # which are invalid regex escapes for pytest.raises(match=...).
        with pytest.raises(
            NotADirectoryError, match=re.escape(f"'{missing}' is not a directory.")
        ):
            visualize_stats(sim, save_path=os.path.join(missing, "quant_stats.html"))

    def test_no_html_extension_error(self):
        """A ValueError is raised if save_path does not end with .html."""
        sim = _tiny_sim()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="'save_path' must end with '.html'."):
                visualize_stats(sim, save_path=os.path.join(tmp_dir, "quant_stats.jpg"))
