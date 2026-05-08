# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for replicating Torch model interface on ONNX InferenceSessions. Borrowed from AI Hub Models"""

import numpy as np
import torch
import onnxruntime
from typing import Iterable, Any, Collection

_TORCH_TO_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

_ORT_TYPE_TO_NP_DTYPE = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}

_NP_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_TO_NP_DTYPE.items()}


def _is_cuda_session(session) -> bool:
    return "CUDAExecutionProvider" in session.get_providers()


def _get_cuda_device_id(session) -> int:
    provider_options = session.get_provider_options()
    cuda_opts = provider_options.get("CUDAExecutionProvider", {})
    return int(cuda_opts.get("device_id", "0"))


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


def _resolve_output_shapes(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
) -> list[tuple[int, ...]] | None:
    """Resolve concrete output shapes from session metadata and input shapes.

    Returns None if any dimension cannot be resolved, allowing the caller to
    fall back to the standard numpy-based inference path.
    """
    input_shapes = {name: tuple(t.shape) for name, t in inputs.items()}
    sym_table: dict[str, int] = {}
    for meta in session.get_inputs():
        for i, dim in enumerate(meta.shape):
            if isinstance(dim, str):
                sym_table[dim] = input_shapes[meta.name][i]

    if "input_ids" in input_shapes:
        seq_len = input_shapes["input_ids"][1]
        batch_size = input_shapes["input_ids"][0]
    elif "inputs_embeds" in input_shapes:
        seq_len = input_shapes["inputs_embeds"][1]
        batch_size = input_shapes["inputs_embeds"][0]
    else:
        seq_len = None
        batch_size = None

    output_shapes = []
    for meta in session.get_outputs():
        shape = []
        for dim_idx, dim in enumerate(meta.shape):
            if isinstance(dim, int):
                shape.append(dim)
            elif isinstance(dim, str) and dim in sym_table:
                shape.append(sym_table[dim])
            elif "past_" in meta.name and "_out" in meta.name and dim_idx == 2:
                in_name = meta.name.replace("_out", "_in").replace("_updated", "")
                if in_name in input_shapes and seq_len is not None:
                    shape.append(input_shapes[in_name][2] + seq_len)
                else:
                    return None
            elif "logits" in meta.name and dim_idx == 0 and batch_size is not None:
                shape.append(batch_size)
            else:
                return None
        output_shapes.append(tuple(shape))
    return output_shapes


def _iobinding_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    device_id = _get_cuda_device_id(session)
    torch_device = f"cuda:{device_id}"
    input_names = [inp.name for inp in session.get_inputs()]

    inputs = kwargs_to_dict(input_names, *args, **kwargs)
    output_shapes = _resolve_output_shapes(session, inputs)

    if output_shapes is None:
        return _numpy_inference(session, inputs, torch_device)

    binding = session.io_binding()
    output_metas = session.get_outputs()

    bound_inputs = []
    for name, tensor in inputs.items():
        tensor = tensor.contiguous()
        bound_inputs.append(tensor)
        binding.bind_input(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=_TORCH_TO_NP_DTYPE[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    output_tensors = []
    for meta, shape in zip(output_metas, output_shapes):
        np_dtype = _ORT_TYPE_TO_NP_DTYPE[meta.type]
        tensor = torch.empty(
            shape, dtype=_NP_TO_TORCH_DTYPE[np_dtype], device=torch_device
        )
        output_tensors.append(tensor)
        binding.bind_output(
            name=meta.name,
            device_type="cuda",
            device_id=device_id,
            element_type=np_dtype,
            shape=shape,
            buffer_ptr=tensor.data_ptr(),
        )

    session.run_with_iobinding(binding)

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


def _numpy_inference(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
    target_device: str,
) -> torch.Tensor | Collection[torch.Tensor]:
    """Fallback: run via numpy when output shapes cannot be resolved."""
    np_inputs = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
    output_np = session.run(None, np_inputs)
    output_tensors = [torch.from_numpy(out).to(target_device) for out in output_np]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


def mock_torch_onnx_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    if _is_cuda_session(session):
        return _iobinding_inference(session, *args, **kwargs)

    input_names = [inp.name for inp in session.get_inputs()]
    inputs = kwargs_to_dict(input_names, *args, **kwargs)
    return _numpy_inference(session, inputs, "cpu")


def _flatten_tensor_args(args):
    """Flatten any tensor lists in positional args for ONNX session input."""
    flat = []
    for a in args:
        if isinstance(a, list) and a and isinstance(a[0], torch.Tensor):
            flat.extend(a)
        else:
            flat.append(a)
    return tuple(flat)


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
        QuantSim forward pass with torch.Tensor.

        Tensor lists in positional args (e.g. deepstack_visual_embeds) are
        flattened to match the ONNX graph's flat input layout.
        """
        assert self.quantsim is not None
        flat_args = _flatten_tensor_args(args)
        return mock_torch_onnx_inference(self.quantsim.session, *flat_args, **kwargs)
