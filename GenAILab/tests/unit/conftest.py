# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for GenAILab unit tests."""

import pytest

from GenAILab.shared.helpers.yaml_config_parser import YAMLConfigParser


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save and restore the global registry state around each test.

    Prevents registration conflicts when torch and onnx test modules
    both import their respective model classes in the same pytest session.
    """
    saved = {
        "default_llm": YAMLConfigParser._default_llm_cls,
        "model": dict(YAMLConfigParser.model_lookup),
        "adaptation": dict(YAMLConfigParser.adaptation_lookup),
    }
    yield
    YAMLConfigParser._default_llm_cls = saved["default_llm"]
    YAMLConfigParser.model_lookup = saved["model"]
    YAMLConfigParser.adaptation_lookup = saved["adaptation"]
