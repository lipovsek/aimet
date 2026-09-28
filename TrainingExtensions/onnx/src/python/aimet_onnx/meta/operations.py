# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX Operation class and utilities"""

from typing import Dict, List, Tuple, Optional
from aimet_onnx.common.connected_graph.operation import Op as _Op
from aimet_onnx.meta.product import Product


class Op(_Op):
    """Subclass Op inherited from aimet_onnx.common.connected_graph.operation.Op"""

    _parameters: Dict[str, Tuple[Product, Optional[str]]]
    _subgraph_ops: List["Op"]

    def __init__(
        self,
        name: str,
        dotted_name: str,
        output_shape,
        is_anonymous: bool,
        op_type: str,
        domain: str = "",
    ):
        """
        Initializer for Op
        :param name: name of the operation
        :param dotted_name: dotted name of the operation
        :param output_shape: shape of the output product of the operation
        :param is_anonymous: whether this is an anonymous operation
        :param op_type: op_type of the operation
        :param domain: domain of the operation
        """
        super().__init__(name, dotted_name, output_shape, is_anonymous, op_type)
        self._parameters = {}
        self.transposed_params = False
        self.domain = domain
        self._subgraph_ops = []

    def add_param(self, param: str, product: Product, product_type: Optional[str]):
        """Add a parameter product to parameters dictionary"""
        self._parameters[param] = (product, product_type)

    @property
    def parameters(self) -> Dict[str, Tuple[Product, Optional[str]]]:
        """returns parameters of the op"""
        return self._parameters

    @property
    def subgraph_ops(self) -> List["Op"]:
        """
        Returns the ops this op runs inside its own body, in the order they run.
        A control-flow op carries its body as a graph-valued attribute.
        """
        return self._subgraph_ops

    def add_subgraph_op(self, op: "Op"):
        """Record that ``op`` belongs to one of this op's subraphs"""
        self._subgraph_ops.append(op)
