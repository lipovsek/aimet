# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, cast

import torch
from torch import nn
from onnx import defs
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import get_const_value
from onnx2torch.node_converters.registry import (
    OperationDescription,
    _CONVERTER_REGISTRY,
)


class OnnxMatmul(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.matmul(x, y)


# disable_existing_matmul
operation_types_to_disable = {"MatMul": [1, 9, 14]}
domain = defs.ONNX_DOMAIN

for op_type, val in operation_types_to_disable.items():
    for version in val:
        try:
            version = defs.get_schema(
                op_type,
                domain=domain,
                max_inclusive_version=version,
            ).since_version
        except (RuntimeError, defs.SchemaError):
            pass

    description = OperationDescription(
        domain=domain,
        operation_type=op_type,
        version=version,
    )
    if description in _CONVERTER_REGISTRY:
        del _CONVERTER_REGISTRY[description]


@add_converter(operation_type="MatMul", version=13)
@add_converter(operation_type="MatMul", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if node.input_values[1] in graph.initializers:
        weights = graph.initializers[node.input_values[1]].to_torch().T
        in_features, out_features = weights.shape[1], weights.shape[0]
        torch_module = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=None,
        )

        with torch.no_grad():
            torch_module.weight.data = weights

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(
                inputs=(node.input_values[0],),
                outputs=node.output_values,
            ),
        )

    return OperationConverterResult(
        torch_module=OnnxMatmul(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


class OnnxReduceMean(nn.Module, OnnxToTorchModule):
    """ReduceMean for opset 18+ where axes is an input rather than an attribute."""

    def __init__(
        self,
        axes: Optional[List[int]] = None,
        keepdims: int = 1,
        noop_with_empty_axes: int = 0,
    ):
        super().__init__()
        self.axes = axes
        self.keepdims = bool(keepdims)
        self.noop_with_empty_axes = noop_with_empty_axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.axes is None or len(self.axes) == 0:
            if self.noop_with_empty_axes:
                return input_tensor
            axes = list(range(input_tensor.dim()))
        else:
            axes = self.axes
        return torch.mean(input_tensor, dim=axes, keepdim=self.keepdims)


@add_converter(operation_type="ReduceMean", version=18)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    keepdims: int = node.attributes.get("keepdims", 1)
    noop_with_empty_axes: int = node.attributes.get("noop_with_empty_axes", 0)

    axes = None
    if len(node.input_values) == 2:
        try:
            axes = cast(torch.Tensor, get_const_value(node.input_values[1], graph))
            axes = axes.tolist()
        except KeyError:
            pass

    return OperationConverterResult(
        torch_module=OnnxReduceMean(
            axes=axes,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        ),
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )


class OnnxReshapeAllowZero(nn.Module, OnnxToTorchModule):
    """Reshape with allowzero=1: zeros in shape are kept as-is (not inherited from input)."""

    def forward(self, input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        return torch.reshape(input_tensor, torch.Size(shape))


# Override the existing Reshape converter to support allowzero=1
_reshape_description = OperationDescription(
    domain=defs.ONNX_DOMAIN,
    operation_type="Reshape",
    version=14,
)
if _reshape_description in _CONVERTER_REGISTRY:
    del _CONVERTER_REGISTRY[_reshape_description]


@add_converter(operation_type="Reshape", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    from onnx2torch.node_converters.reshape import OnnxReshape

    if node.attributes.get("allowzero", 0) == 1:
        return OperationConverterResult(
            torch_module=OnnxReshapeAllowZero(),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )

    return OperationConverterResult(
        torch_module=OnnxReshape(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type="Constant", version=19)
@add_converter(operation_type="Constant", version=21)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    from onnx2torch.node_converters.constant import OnnxConstant, _prepare_output_value

    attr_name, value = list(node.attributes.items())[0]
    prepared_value = _prepare_output_value(value, attr_name)

    return OperationConverterResult(
        torch_module=OnnxConstant(value=prepared_value),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


class OnnxScatterElements(nn.Module, OnnxToTorchModule):
    """ScatterElements: scatter updates into data at indices along axis."""

    # ONNX reduction -> torch.scatter_reduce reduce mode.
    _REDUCE_MAP = {"add": "sum", "mul": "prod", "max": "amax", "min": "amin"}

    def __init__(self, axis: int = 0, reduction: str = "none"):
        super().__init__()
        if reduction != "none" and reduction not in self._REDUCE_MAP:
            raise NotImplementedError(
                f"ScatterElements reduction {reduction!r} is not supported."
            )
        self.axis = axis
        self.reduction = reduction

    def forward(
        self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor
    ) -> torch.Tensor:
        if self.reduction == "none":
            return torch.scatter(data, self.axis, indices, updates)
        return data.scatter_reduce(
            self.axis,
            indices,
            updates,
            reduce=self._REDUCE_MAP[self.reduction],
            include_self=True,
        )


@add_converter(operation_type="ScatterElements", version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis: int = node.attributes.get("axis", 0)
    reduction: str = node.attributes.get("reduction", "none")
    return OperationConverterResult(
        torch_module=OnnxScatterElements(axis=axis, reduction=reduction),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


# Bulk-register missing op versions for opsets 18-21.
# Many ops have unchanged schemas at newer versions but onnx2torch only registers
# up to opset 13-17. We find the latest registered converter for each op and
# re-register it at the newer since_version.
def _register_missing_opset_converters(max_opset: int = 21):
    """Register converters for op versions up to max_opset using existing converters."""
    _domain = defs.ONNX_DOMAIN

    # Build map of (domain, op_type) -> max registered version
    _registered = {}
    for desc in list(_CONVERTER_REGISTRY.keys()):
        key = (desc.domain, desc.operation_type)
        if key not in _registered or desc.version > _registered[key]:
            _registered[key] = desc.version

    for (d, op), max_ver in _registered.items():
        if d != _domain:
            continue
        for opset in range(max_ver + 1, max_opset + 1):
            try:
                schema = defs.get_schema(op, opset, _domain)
                since = schema.since_version
            except (RuntimeError, defs.SchemaError):
                continue
            if since > max_ver:
                desc_new = OperationDescription(
                    domain=_domain, operation_type=op, version=since
                )
                if desc_new not in _CONVERTER_REGISTRY:
                    # Reuse the converter from the latest registered version
                    desc_old = OperationDescription(
                        domain=_domain, operation_type=op, version=max_ver
                    )
                    _CONVERTER_REGISTRY[desc_new] = _CONVERTER_REGISTRY[desc_old]


_register_missing_opset_converters(max_opset=21)
