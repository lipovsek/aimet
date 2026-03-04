# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Pytorch Operation class and utilities"""

from typing import Union
import torch

from aimet_torch.common.connected_graph.operation import Op as _Op


class Op(_Op):
    """Subclass Op inherited from aimet_torch.common.connected_graph.operation.Op"""

    def __init__(
        self,
        name: str,
        dotted_name: str,
        output_shape,
        is_anonymous: bool,
        op_type: str,
        residing_module: Union[torch.nn.Module, None],
    ):
        """
        Initializer for Op
        :param name: name of the operation
        :param dotted_name: dotted name of the operation
        :param output_shape: shape of the output product of the operation
        :param is_anonymous: whether this is an anonymous operation
        :param op_type: type of the operation
        :param residing_module: Torch module in which the op is situated
        """
        super().__init__(name, dotted_name, output_shape, is_anonymous, op_type)
        self.residing_module = residing_module

    def is_grid_preserving_op(self) -> bool:
        from .connectedgraph import ConnectedGraph
        from aimet_torch.common.onnx._utils import _is_grid_preserving_op
        from ..onnx_utils import map_torch_types_to_onnx
        from ..nn import QuantizationMixin

        module = self.get_module()

        if not module:
            return (
                self.type == "CG_Split"
                or self.type in ConnectedGraph.math_invariant_types
            )

        if isinstance(module, QuantizationMixin):
            module = module.get_original_module()

        module_cls = type(module)
        onnx_op_types = map_torch_types_to_onnx.get(module_cls)

        if not onnx_op_types:
            # ONNX op type unknown.
            # To be safe, we should assume non-grid-preserving op in this case.
            return False

        return all(_is_grid_preserving_op(op_type) for op_type in onnx_op_types)
