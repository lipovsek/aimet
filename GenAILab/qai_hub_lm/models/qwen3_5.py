# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Qwen-3.5 model mixin for Torch and ONNX frameworks."""


class Qwen3_5Mixin:
    @staticmethod
    def use_dynamo_export() -> bool:
        # Qwen 3.5 must export via dynamo rather than torch.jit.trace for two
        # independent reasons, neither tied to the chosen sequence length:
        #   1. The model's forward builds its attention mask with
        #      ``create_causal_mask`` (data-dependent control flow), which the
        #      TorchScript tracer cannot capture.
        #   2. Tracing only follows the prefill branch (use_precomputed_states
        #      is False), which hardcodes initial_state=None for the linear
        #      attention layers. That makes the recurrent-state inputs dead
        #      code, so torchscript prunes them from the graph. At inference the
        #      Generator sends all inputs positionally, causing a shape
        #      mismatch. Dynamo preserves all input placeholders regardless of
        #      use.
        return True
