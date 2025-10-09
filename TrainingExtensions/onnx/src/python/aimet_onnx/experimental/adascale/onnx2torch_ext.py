# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn

from onnx import defs, numpy_helper
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
from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)

from aimet_onnx.experimental.adascale.quantizer import QuantizedLinear


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

torch_to_onnx_mapping = {}
node_name_mapping = {}


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

        torch_to_onnx_mapping[torch_module] = {
            "weight": node.input_values[1],
        }
        node_name_mapping[OnnxGraph.generate_node_name(node)] = (
            torch_module,
            node.name,
            node.input_values[1],
        )

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


def copy_pt_weights_to_onnx(pt_block, onnx_model):
    """
    Given a pt_block with adascale params computed, copy the params to onnx model
    """
    initializer_name_to_index_map = {
        init.name: idx for idx, init in enumerate(onnx_model.graph.initializer)
    }
    for name, module in pt_block.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        pytorch_weight = (
            module.param_quantizers["weight"]
            .get_folded_weight(module.weight)
            .detach()
            .cpu()
            .numpy()
        )

        onnx_tensor_name = node_name_mapping[name][2]
        onnx_param_tensor = numpy_helper.to_array(
            onnx_model.graph.initializer[
                initializer_name_to_index_map[onnx_tensor_name]
            ]
        )
        pytorch_weight = pytorch_weight.T
        if pytorch_weight.shape != onnx_param_tensor.shape:
            raise ValueError(
                f"pt param shape {pytorch_weight.shape} did not match onnx shape {onnx_param_tensor.shape}"
            )
        if not (pytorch_weight == onnx_param_tensor).all():
            onnx_model.graph.initializer[
                initializer_name_to_index_map[onnx_tensor_name]
            ].CopyFrom(numpy_helper.from_array(pytorch_weight, onnx_tensor_name))
            _logger.info(
                "Copy from PyTorch to ONNX: torch : %s  onnx param : %s",
                name,
                onnx_tensor_name,
            )
