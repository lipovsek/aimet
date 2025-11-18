# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Main class for pattern match based graph searcher"""

from typing import Callable, Optional, List, Set
from .connected_graph.connectedgraph import ConnectedGraph
from .graph_pattern_matcher import PatternType
from .utils import AimetLogger
from .connected_graph.operation import Op

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


# TODO: #5597: Remove Conv3d and Depthwise Conv supergroup once HTP support is added
def _check_if_conv3d(op: Op) -> bool:
    if op.type != "Conv" and op.type != "ConvTranspose":
        return False

    if op.inputs[0].shape is not None and len(op.inputs[0].shape) == 5:
        return True

    # Additional check on weights shape if available.
    if op.inputs[1].shape is not None and len(op.inputs[1].shape) == 5:
        return True

    # Not a conv3d op
    return False


# TODO: #5597: Remove Conv3d and Depthwise Conv supergroup once HTP support is added
def _check_if_depthwise_conv(op: Op) -> bool:
    if op.type != "Conv" and op.type != "ConvTranspose":
        return False

    if not hasattr(op, "groups") or op.groups == 1:
        return False

    if len(op.inputs) < 2:
        raise RuntimeError("Expecting at least two inputs to Conv op.")

    input_shape = op.inputs[0].shape
    weight_shape = op.inputs[1].shape

    groups = op.groups

    # depthwise_conv: each channel in it's own group
    if input_shape is None or groups != input_shape[1]:
        return False

    # additional validation
    if weight_shape is not None and op.type == "Conv" and weight_shape[1] != 1:
        # For Depthwise Conv, weight shape[1] should be 1
        return False
    elif (
        weight_shape is not None
        and len(op.outputs) > 1
        and op.outputs[0].shape is not None
        and op.type == "ConvTranspose"
        and weight_shape[1] != op.outputs[0].shape[1] / groups
    ):
        # For Depthwise Deconv, weight shape[1] should be output_channels / groups
        return False

    # Indeed depthwise Conv/Deconv
    return True


# TODO: #5597: Remove Conv3d and Depthwise Conv supergroup once HTP support is added
def _check_if_conv3d_or_depthwise_conv(op: Op) -> bool:
    if _check_if_conv3d(op):
        return True
    if _check_if_depthwise_conv(op):
        return True
    return False


class GraphSearcher:
    """
    Graph searcher class performs graph search on connected graph.
    It uses SlidingWindow to maintain the search window and PatternMatcher to match sub graph patterns.
    """

    def __init__(
        self, conn_graph: ConnectedGraph, patterns_with_callback: List[PatternType]
    ):
        """
        initializes params required for pattern matching
        :param patterns_with_callback: patterns with corresponding call back functions
        """
        self._connected_graph = conn_graph
        self._patterns_with_callbacks = patterns_with_callback
        self.type_to_op_dict = {}
        for op in conn_graph.get_all_ops().values():
            if op.type in self.type_to_op_dict:
                self.type_to_op_dict[op.type].append(op)
            else:
                self.type_to_op_dict[op.type] = [op]

        self._already_matched: Set[Op] = set()

    # pylint: disable=too-many-nested-blocks
    def find_all_patterns_in_graph_apply_actions(
        self,
        ignore: Optional[Op] = None,
        op_pattern_to_reject: Callable[[Op], bool] = None,
        disjoint: bool = False,
    ):
        """
        Find corresponding op sequences and apply actions.
        :param ignore: List of operations to ignore during searching
        :param op_pattern_to_reject: Callable to perform additional checks on Op to reject pattern match.
            This is useful to express intent on patterns that should not be matched.
            Since GraphSearcher performs high level pattern match, this enables to provide override for aggressive rejection for a given op config.
        :param disjoint: If True, ensures that matched patterns do not share any ops.
        """

        if ignore is None:
            ignore = []

        # Search patterns starting with longer patterns first
        for pattern_type in sorted(
            self._patterns_with_callbacks, key=lambda l: len(l.pattern), reverse=True
        ):
            if pattern_type.pattern[0] in self.type_to_op_dict:
                # One or more ops in the graph correspond to the current pattern's starting op type
                for op in self.type_to_op_dict[pattern_type.pattern[0]]:
                    matched_ops = self._match_pattern(
                        op, pattern_type.pattern, ignore, op_pattern_to_reject
                    )
                    if not matched_ops:
                        continue
                    for matched_ops_list in matched_ops:
                        if disjoint and any(
                            matched_op in self._already_matched
                            for matched_op in matched_ops_list
                        ):
                            # This pattern has already been matched as part of a longer pattern
                            continue
                        else:
                            self._already_matched |= set(matched_ops_list)

                        pattern_type.action(pattern_type, matched_ops_list)
                        logger.debug("found match: %s", matched_ops_list)

    # pylint: disable=too-many-branches, too-many-return-statements
    def _match_pattern(
        self,
        op: Op,
        pattern: List[str],
        ignored_ops: List[Op],
        op_pattern_to_reject: Optional[Callable[[Op], bool]] = None,
    ) -> Optional[List[List[Op]]]:
        if not pattern:
            return []

        matched_ops = None
        if op in ignored_ops:
            if not op.outputs:
                return None
            for child_op in op.output_ops:
                matched_child_ops = self._match_pattern(child_op, pattern, ignored_ops)
                if matched_child_ops is not None:
                    if matched_ops is None:
                        matched_ops = []
                    matched_ops.extend(matched_child_ops)
            return matched_ops

        if op.type != pattern[0]:
            return None

        # If additional op pattern checks are provided and matches for rejection, early exit.
        # This is useful to provide more aggresive checks e.g. depthwise conv or 3d conv, ...

        if op_pattern_to_reject is not None and op_pattern_to_reject(op):
            return False

        if len(pattern) > 1:
            # Still more to match
            if not op.outputs:
                return None
            if len(op.output_ops) > 1:  # Can't match patterns with branches
                return None
            for child_op in op.output_ops:
                matched_child_ops = self._match_pattern(
                    child_op, pattern[1:], ignored_ops
                )
                if matched_child_ops:
                    if matched_ops is None:
                        matched_ops = []
                    for matched_child_op_list in matched_child_ops:
                        matched_ops.append([op] + matched_child_op_list)
        else:
            matched_ops = [[op]]

        return matched_ops
