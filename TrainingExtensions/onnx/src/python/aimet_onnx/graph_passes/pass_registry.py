# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=missing-module-docstring

from aimet_onnx.graph_passes.graph_pass import GraphPass
from typing import Dict, List, TypeVar, Generic
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.utils import ModelProto
from ..meta.operations import Op


PassType = TypeVar("PassType")


class BaseRegistry(Generic[PassType]):
    """
    Registry for Graph passes.
    """

    def __init__(self):
        self.passes: Dict[str, PassType] = {}

    def __getitem__(self, name: str) -> PassType:
        """
        return Graph Pass class associated with given pass name
        """
        if name in self.passes:
            return self.passes[name]
        raise KeyError(f"Pass {name} not found.")

    def __contains__(self, name: str) -> bool:
        """
        Check if given pass is registered.
        """
        return name in self.passes

    def register(self, pass_cls: PassType, name: str, override: bool = False):
        """
        Register Graph Pass

        Args:
            pass_cls (PassType): GraphPass class being registered.
            name (str): Pass name to register graph pass with.
            override (bool, optional): Override existing pass if set. Defaults to False.

        Raises:
            RuntimeError: If override is not set and GraphPass already exists.
        """
        if name in self.passes and not override:
            raise RuntimeError(
                f"Pass {name} is already registered. Consider using override if intent to replace default pass."
            )
        self.passes[name] = pass_cls


class PassRegistry(BaseRegistry[GraphPass]):
    """
    Registry for Graph passes.
    """

    def __getitem__(self, name: str) -> GraphPass:
        """
        return Graph Pass object associated with given pass name
        """
        return super().__getitem__(name)()


# Global Pass Registry to hold all graph passes
PASS_REGISTRY = PassRegistry()


def register_pass(name: str, override: bool = False):
    """
    Decorate to register graph pass

    Args:
        name (str): Pass name to register graph pass with.
        override (bool, optional): Override pass if already registered. Defaults to False.
    """

    def wrapper(pass_cls: GraphPass):
        PASS_REGISTRY.register(pass_cls, name, override)
        return pass_cls

    return wrapper


def apply_graph_passes(
    model: ModelProto,
    connected_graph: ConnectedGraph,
    op_to_quantizers: Dict[str, QcQuantizeOp],
    passes_to_run: List[str],
):
    """
    Runs list of graph passes on input ConnectedGraph

    Args:
        connected_graph (ConnectedGraph): Input graph to run graph passes on
        op_to_quantizers (Dict[str, QcQuantizeOp]): Global map of Quantization ops.
        passes_to_run (List[str]): List of graph passes to run.

    Raises:
        ValueError: If requested GraphPass does not exists.
    """
    for p in passes_to_run:
        if p in PASS_REGISTRY:
            PASS_REGISTRY[p](model, connected_graph, op_to_quantizers)
        else:
            raise ValueError(f"Graph pass requested but not found: {p}")


def find_all_matches(
    model: ModelProto,
    connected_graph: ConnectedGraph,
    passes_to_run: List[str],
) -> List[List[Op]]:
    """
    Runs list of graph passes on input ConnectedGraph

    Args:
        connected_graph (ConnectedGraph): Input graph to run graph passes on
        passes_to_run (List[str]): List of graph passes to run.

    Raises:
        ValueError: If requested GraphPass does not exists.
    """
    matches = []

    for p in passes_to_run:
        for op in connected_graph.ordered_ops:
            if p in PASS_REGISTRY:
                graph_pass = PASS_REGISTRY[p]
                match = graph_pass.match_pattern(op, model)
                if match:
                    matches.append(match)
            else:
                raise ValueError(f"Graph pass requested but not found: {p}")

    return matches
