# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-3.5 model class"""

from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.qwen3_5 import Qwen3_5Mixin


@YAMLConfigParser.register_model("qwen3_5")
class Qwen3_5(Qwen3_5Mixin, LLM_ONNX):
    pass
