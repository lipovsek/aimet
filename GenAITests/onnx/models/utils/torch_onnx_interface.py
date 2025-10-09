# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for replicating Torch model interface on ONNX InferenceSessions. Borrowed from AI Hub Models"""

import torch
import onnxruntime
from typing import Iterable, Any, Collection


def kwargs_to_dict(argnames: Iterable[str], *args, **kwargs) -> dict[str, Any]:
    input_dict: dict[str, Any] = dict()
    for idx, input_name in enumerate(argnames):
        if len(args) > idx:
            input_val = args[idx]
            if input_name in kwargs:
                raise ValueError(
                    f"Cannot pass input {input_name} twice (as a positional arg and a keyword arg)."
                )
        elif input_name in kwargs:
            input_val = kwargs[input_name]
        else:
            raise ValueError(f"Missing input {input_name}")
        input_dict[input_name] = input_val
    return input_dict


def mock_torch_onnx_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    input_names = [inp.name for inp in session.get_inputs()]

    inputs = {
        k: v.cpu().detach().numpy()
        for k, v in kwargs_to_dict(input_names, *args, **kwargs).items()
    }
    output_np = session.run(None, inputs)
    output_tensors = [torch.from_numpy(out) for out in output_np]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


class TorchONNXInterface(torch.nn.Module):
    def __init__(self, quantsim, config):
        super().__init__()
        self.quantsim = quantsim
        self._config = config

    @property
    def config(self):
        return self._config

    @property
    def device(self) -> torch.device:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """
        QuantSim forward pass with torch.Tensor
        """
        assert self.quantsim is not None
        return mock_torch_onnx_inference(self.quantsim.session, *args, **kwargs)
