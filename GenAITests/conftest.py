# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""pytest config for test_genai.py"""

import pytest

from GenAITests.shared.helpers.fp_cache import DiskBackedFPCache
from GenAITests.shared.helpers.model_cache import DiskBackedModelCache
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser


def pytest_addoption(parser):
    parser.addoption("--config", action="store", default=None)
    parser.addoption("--fp-cache-dir", action="store", default=".fp_cache")
    parser.addoption("--clear-fp-cache", action="store_true", default=False)
    parser.addoption("--model-cache-dir", action="store", default=".model_cache")
    parser.addoption("--clear-model-cache", action="store_true", default=False)


@pytest.fixture(scope="session")
def fp_cache(request):
    """Session-scoped disk-backed FP results cache.

    Shared across all parameterized test cases so that FP model outputs are
    computed at most once per (model, dataset) combination, even across
    different quantization recipes in a multi-doc config.
    """
    cache = DiskBackedFPCache(request.config.getoption("--fp-cache-dir"))
    if request.config.getoption("--clear-fp-cache"):
        cache.clear()
    return cache


@pytest.fixture(scope="session")
def model_cache(request):
    """Session-scoped disk-backed ONNX model cache.

    Shared across all parameterized test cases so that ONNX model exports are
    computed at most once per (model, sequence_length, context_length, adaptations)
    combination, even across different quantization recipes in a multi-doc config.
    """
    cache = DiskBackedModelCache(request.config.getoption("--model-cache-dir"))
    if request.config.getoption("--clear-model-cache"):
        cache.clear()
    return cache


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("--config", skip=False)

    test_parameters = (
        list(YAMLConfigParser.parse(config_file)) if config_file else [None]
    )
    if "test_parameters" in metafunc.fixturenames:
        # Generate test cases based on the test parameters list from the config file
        metafunc.parametrize("test_parameters", test_parameters)
