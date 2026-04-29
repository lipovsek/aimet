# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for GenAILab unit tests."""

import pytest

from GenAILab.bench.yaml_config_parser import YAMLConfigParser

_orig_register_llm = YAMLConfigParser.register_default_llm
_orig_register_model = YAMLConfigParser.register_model


def pytest_configure(config):
    """Allow both ONNX and Torch frameworks to register during test collection.

    Registration decorators fire at module import time. When pytest collects
    both ONNX and Torch test files, both frameworks' modules get imported,
    and the second registration would normally raise. Patching the guards
    to silently overwrite avoids collection-time crashes.
    """

    @classmethod
    def _permissive_register_llm(cls, llm_cls):
        cls._default_llm_cls = llm_cls
        return llm_cls

    @classmethod
    def _permissive_register_model(cls, model_type):
        def decorator(model_cls):
            cls.model_lookup[model_type] = model_cls
            return model_cls

        return decorator

    YAMLConfigParser.register_default_llm = _permissive_register_llm
    YAMLConfigParser.register_model = _permissive_register_model


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Reset registry state and restore strict methods before each test."""
    YAMLConfigParser.register_default_llm = _orig_register_llm
    YAMLConfigParser.register_model = _orig_register_model
    YAMLConfigParser._default_llm_cls = None
    YAMLConfigParser.model_lookup = {}
    YAMLConfigParser.adaptation_lookup = {}
    yield
    YAMLConfigParser._default_llm_cls = None
    YAMLConfigParser.model_lookup = {}
    YAMLConfigParser.adaptation_lookup = {}
