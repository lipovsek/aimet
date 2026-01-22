# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utility for checking model architechture prior to using AIMET feature."""

from typing import Callable, Union, Tuple
import torch

from aimet_torch.common.utils import AimetLogger
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.arch_checker.arch_checker_rules import TorchActivations
from aimet_torch.arch_checker.arch_checker_utils import (
    ArchCheckerReport,
    OpStructure,
    check_type_deco,
    get_node_check_dict,
    get_pattern_check_list,
    check_module,
)

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


class ArchChecker:
    """
    ArchChecker object to check pytorch model architechture and suggest better architechture prior
    to training.
    """

    _node_check_dict = get_node_check_dict()
    _pattern_checks = get_pattern_check_list()
    _arch_checker_report = ArchCheckerReport()

    @staticmethod
    def add_node_check(check_target_type: torch.nn.Module, arch_check: Callable):
        """
        Add extra checks for node checks in architecture checker.
        :param check_target_type: layer type to be checked.
        :param arch_check: node checker function.
        """

        # All TorchActivations are included to one type.
        if isinstance(check_target_type, TorchActivations):
            # Init TorchActivations check if not exist.
            if TorchActivations not in ArchChecker._node_check_dict:
                ArchChecker._node_check_dict[TorchActivations] = []

            if check_target_type is TorchActivations:
                ArchChecker._node_check_dict[TorchActivations].append(arch_check)
            else:
                # Add check_type_deco wrapper if check_target_type is a subclass of TorchActivations.
                ArchChecker._node_check_dict[TorchActivations].append(
                    check_type_deco(check_target_type)(arch_check)
                )

        else:
            if check_target_type in ArchChecker._node_check_dict:
                ArchChecker._node_check_dict[check_target_type].append(arch_check)
            else:
                ArchChecker._node_check_dict[check_target_type] = [arch_check]

    @staticmethod
    def add_pattern_check(arch_check: Callable):
        """
        Add extra checks for pattern checks in architecture checker.
        :param arch_check: pattern checker function.
        :param check_kind: Check kind. Different check kind requires different arguments.
        """
        ArchChecker._pattern_checks.append(arch_check)

    @staticmethod
    def check_model_arch(
        model: torch.nn.Module,
        dummy_input: Union[torch.Tensor, Tuple],
        result_dir: str = None,
    ) -> ArchCheckerReport:
        """

        Check each node in the model using checks in _node_check_dict. Record only the nodes and
        failed tests.

        :param model: Torch model to be checked.
        :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
        :return arch_checker_report: {op.dotted_name_op: NodeErrorReportObject }
        """

        def run_node_checks():
            """
            Walk through connected_graph and applies node checks on each node.
            """
            for op in connected_graph.ordered_ops:
                module = op.get_module()
                if module and isinstance(
                    module, tuple(ArchChecker._node_check_dict.keys())
                ):
                    if isinstance(module, TorchActivations):
                        checks = ArchChecker._node_check_dict[TorchActivations]
                    else:
                        checks = ArchChecker._node_check_dict[type(module)]

                    failed_checks_set = check_module(module, checks)
                    if failed_checks_set:
                        ArchChecker._arch_checker_report.update_raw_report(
                            op, failed_checks_set
                        )
                        logger.info(
                            "Graph/Node: %s: %s fails check: %s",
                            op.dotted_name,
                            module,
                            failed_checks_set,
                        )

        def run_patten_check():
            """
            Applies pattern checks on connected graph.
            """
            for _check in ArchChecker._pattern_checks:
                failed_check_ops = _check(connected_graph)

                if failed_check_ops:
                    # Pattern check that marks structure returns List[List[Op]]
                    # Transform List[List[Op]] to List[OpStructure]
                    if isinstance(failed_check_ops[0], list):
                        failed_check_ops = [
                            OpStructure(_op_tuple) for _op_tuple in failed_check_ops
                        ]

                    ArchChecker._arch_checker_report.update_raw_report(
                        failed_check_ops, _check.__name__
                    )
                    for op in failed_check_ops:
                        logger.info(
                            "Graph/Node: %s: %s fails check: %s",
                            op.dotted_name,
                            op.get_module(),
                            {_check.__name__},
                        )

        connected_graph = ConnectedGraph(model, dummy_input)
        # Run all node checkes
        logger.info("Running node checkes.")
        run_node_checks()

        # Run all pattern checkes
        logger.info("Running pattern checkes.")
        run_patten_check()

        if result_dir is not None:
            ArchChecker.set_export_dir(result_dir)
        ArchChecker._arch_checker_report.export_to_html()

    @staticmethod
    def set_export_dir(dir_path: str):
        """Set export dir."""
        ArchChecker._arch_checker_report.result_dir = dir_path
