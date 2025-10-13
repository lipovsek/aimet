# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

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

for op, val in operation_types_to_disable.items():
    for version in val:
        try:
            version = defs.get_schema(
                op,
                domain=domain,
                max_inclusive_version=version,
            ).since_version
        except (RuntimeError, defs.SchemaError):
            pass

    description = OperationDescription(
        domain=domain,
        operation_type=op,
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
