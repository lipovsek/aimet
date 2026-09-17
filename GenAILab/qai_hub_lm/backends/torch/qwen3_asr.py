# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen3-ASR Torch backend.

Lives in its own module rather than in ``vlm.py`` because the audio encoder
needs one piece of aimet-side setup before a sim can be built over it (see
below), and ``models/qwen3_asr.py`` is deliberately aimet-free.
"""

from __future__ import annotations

from aimet_torch.v2.nn.true_quant import QuantizationMixin
from transformers.models.qwen3_asr.modeling_qwen3_asr import (
    SinusoidsPositionEmbedding,
)

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.backends.torch.vlm import VLM_Torch
from GenAILab.qai_hub_lm.models.qwen3_asr import Qwen3ASR_LM

# The audio encoder's sinusoidal position table is a fixed, non-learnable buffer,
# and the encoder does not even call the module -- it indexes the buffer directly
# (`self.positional_embedding.positional_embedding[:time_steps]`). There is
# nothing to quantize, but QuantSim still walks every submodule and refuses to
# proceed on a type it has no quantized definition for. Excluding it is the
# documented escape hatch for exactly this case; quantizing a constant table
# would only add error for no benefit.
QuantizationMixin.ignore(SinusoidsPositionEmbedding)


@YAMLConfigParser.register_model("qwen3_asr")
class Qwen3ASR_Torch(VLM_Torch, Qwen3ASR_LM):
    pass
