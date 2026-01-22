# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Product class and utilities"""

from typing import TYPE_CHECKING
from onnx import TensorProto
from aimet_onnx.common.connected_graph.product import Product as _Product

if TYPE_CHECKING:
    from aimet_onnx.meta.operations import Op


class Product(_Product):
    """Subclass Product inherited from aimet_onnx.common.connected_graph.operation.Product"""

    def __init__(self, name, shape):
        super().__init__(name, shape)
        self.tensor_dict = {}
        self.tensor = None

    def set_as_param(self, op: "Op", tensor: TensorProto):
        self.shape = tensor.dims
        self.is_parm = True
        self.tensor_dict[op] = tensor
        self.tensor = tensor
        self.is_const = False  # Backward compatibility
