# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import itertools
import torch
import aimet_torch.v2 as aimet
from aimet_torch.v2.quantsim import QuantizationSimModel
from ..models_ import test_models


def test_compute_param_encodings():
    model = test_models.TinyModel()
    dummy_input = torch.rand(1, 3, 32, 32)

    sim = QuantizationSimModel(model, dummy_input)
    aimet.nn.compute_param_encodings(sim.model)

    for qmodule in model.modules():
        if not isinstance(qmodule, aimet.nn.BaseQuantizationMixin):
            continue

        for q in qmodule.param_quantizers.values():
            assert q.is_initialized()

        for q in itertools.chain(qmodule.input_quantizers, qmodule.output_quantizers):
            assert not q.is_initialized()


def test_encoding_analyzer_cleared_after_computing_param_encodings():
    model = test_models.TinyModel()
    dummy_input = torch.rand(1, 3, 32, 32)

    sim = QuantizationSimModel(model, dummy_input)
    with aimet.nn.compute_encodings(sim.model):
        pass

    for qmodule in sim.model.modules():
        if not isinstance(qmodule, aimet.nn.QuantizationMixin):
            continue
        for q in qmodule.param_quantizers.values():
            if not isinstance(q, aimet.quantization.affine.AffineQuantizerBase):
                continue

            observer_stats = q.encoding_analyzer.observer.get_stats()
            assert observer_stats.min is None and observer_stats.max is None
            assert q.is_initialized()
