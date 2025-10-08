# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import torch

from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.nn.modules.custom import QuantizedHadamardRotation
from aimet_torch._base.nn.modules.custom import HadamardRotation
from aimet_torch.experimental.transforms.transform_ops import (
    InvertibleTransformOp,
    MatrixTransformOp,
)


# pylint: disable=abstract-method
class ScaledRotateTransformOp(InvertibleTransformOp):
    def __init__(self, head_dim, num_attention_heads, num_key_value_heads):
        super().__init__(mergeable=True)

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        k_proj_out_features = head_dim * num_key_value_heads

        self.rotation = torch.nn.Parameter(
            torch.zeros(k_proj_out_features // 2), requires_grad=True
        )
        self.scale = torch.nn.Parameter(
            torch.ones((num_key_value_heads, head_dim // 2)), requires_grad=True
        )

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x * cos + ScaledRotateTransformOp.rotate_half(x) * sin

    def get_rotation(self) -> tuple[torch.Tensor, torch.Tensor]:
        cos = torch.cos(self.rotation).view(self.scale.shape).repeat(1, 2)
        sin = torch.sin(self.rotation).view(self.scale.shape).repeat(1, 2)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        repeated_scale = self.scale.repeat(1, 2)
        scaled_x = x.view(*x.shape[:-1], *repeated_scale.shape) * repeated_scale
        cos, sin = self.get_rotation()
        return self.rotate(scaled_x, cos, sin).view(x.shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        repeated_scale = self.scale.repeat_interleave(
            num_key_value_groups, dim=0
        ).repeat(1, 2)
        scaled_x = x.view(*x.shape[:-1], self.num_attention_heads, -1) / repeated_scale
        cos, sin = self.get_rotation()
        cos = cos.repeat_interleave(num_key_value_groups, dim=0)
        sin = sin.repeat_interleave(num_key_value_groups, dim=0)
        return self.rotate(scaled_x, cos, sin).view(x.shape)

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight.T).T


class ScalingTransformOp(InvertibleTransformOp):
    def __init__(self, intermediate_size):
        super().__init__(True)
        self.scale = torch.nn.Parameter(
            torch.ones(intermediate_size), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight.T).T

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight)


class GroupedHadamardTransformOp(InvertibleTransformOp, HadamardRotation):
    def __init__(self, intermediate_size: int):
        super().__init__(mergeable=False, size=intermediate_size)

    def forward(self, x):
        return HadamardRotation.forward(self, x)

    def inverse(self, x):
        # pylint: disable=access-member-before-definition, attribute-defined-outside-init
        hadamard = self.hadamard

        try:
            self.hadamard = hadamard.transpose(1, 0)
            return HadamardRotation.forward(self, x)
        finally:
            self.hadamard = hadamard

    def get_inverted_op(self):
        inverted_op = super().get_inverted_op()
        inverted_op.mergeable = True
        return inverted_op

    def left_hand_merge(self, weight):
        return self.forward(weight)


@QuantizationMixin.implements(GroupedHadamardTransformOp)
class QuantizedGroupedHadamardTransformOp(QuantizedHadamardRotation):
    pass


class MultiHeadValueTransformOp(InvertibleTransformOp):
    def __init__(self, head_dim, num_attention_heads, num_key_value_heads):
        super().__init__(mergeable=True)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.matrix_per_head = torch.nn.ModuleList(
            [
                MatrixTransformOp(torch.eye(head_dim))
                for _ in range(self.num_key_value_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slices = torch.chunk(x, self.num_key_value_heads, dim=-1)
        results = [
            transform(slice) for transform, slice in zip(self.matrix_per_head, slices)
        ]
        return torch.cat(results, dim=-1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        slices = torch.chunk(x, self.num_attention_heads, dim=-1)
        repeated_matrices = [
            matrix
            for matrix in self.matrix_per_head
            for _ in range(len(slices) // len(self.matrix_per_head))
        ]
        results = [
            transform.inverse(slice)
            for transform, slice in zip(repeated_matrices, slices)
        ]
        return torch.cat(results, dim=-1)

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.data.dtype
        return (
            self.forward(
                torch.eye(
                    self.head_dim * self.num_attention_heads,
                    device=weight.data.device,
                    dtype=torch.float32,
                )
            )
            @ weight.T.to(dtype=torch.float32)
        ).T.to(dtype=orig_dtype)

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.data.dtype
        return self.forward(weight.T.to(dtype=torch.float32)).T.to(orig_dtype)


class RotationTransformOp(InvertibleTransformOp):
    def __init__(self, matrix: torch.Tensor):
        super().__init__(mergeable=True)
        linear = torch.nn.Linear(*matrix.shape, bias=False)
        linear.weight.data = matrix
        self.rotation = torch.nn.utils.parametrizations.orthogonal(
            linear, orthogonal_map="cayley", use_trivialization=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                f"{type(self)} cannot be exported to ONNX as it involves "
                "complex operations that aren't supported in ONNX. "
                "Please try merging this transformation statically "
                "with 'merge_transforms(True)' before exporting to ONNX"
            )
        orig_dtype = x.dtype
        return (x.to(dtype=torch.float32) @ self.rotation.weight.T).to(dtype=orig_dtype)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                f"{type(self)} cannot be exported to ONNX as it involves "
                "complex operations that aren't supported in ONNX. "
                "Please try merging this transformation statically "
                "with 'merge_transforms(True)' before exporting to ONNX"
            )
        orig_dtype = x.dtype
        return (x.to(dtype=torch.float32) @ self.rotation.weight).to(dtype=orig_dtype)

    def left_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.dtype
        return (
            self.forward(
                torch.eye(
                    self.rotation.weight.data.shape[-1],
                    device=weight.data.device,
                    dtype=torch.float32,
                )
            )
            @ weight.T.to(dtype=torch.float32)
        ).T.to(dtype=orig_dtype)

    def right_hand_merge(self, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = weight.dtype
        return self.forward(weight.T.to(dtype=torch.float32)).T.to(dtype=orig_dtype)


def set_export_to_custom_hadamard(model: torch.nn.Module, enable: bool):
    """
    Enable/disable the behavior of GroupedHadamardTransformOps in model to export to custom qti_aisw HadamardTransform ONNX op.

    :param model: Model to set behavior of
    :param enable: True to enable exporting to custom ONNX op, False otherwise
    """
    for module in model.modules():
        if isinstance(module, GroupedHadamardTransformOp):
            module.use_custom_hadamard = enable
