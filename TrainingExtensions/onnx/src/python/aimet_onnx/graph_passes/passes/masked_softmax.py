# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
from aimet_onnx.graph_passes.graph_pass import SupergroupGraphPass
from aimet_onnx.graph_passes.pass_registry import register_pass


@register_pass("MaskedSoftmax")
class MaskedSoftmax(SupergroupGraphPass):
    """
    Dummy placeholder to bypass quantsim configurator which
    expects every entry of supergroup_list to be registered as SupergroupGraphPass
    even if they are registered as FusionPassRegistry instead.
    """

    # pylint: disable=unused-argument
    def match_pattern(self, *args, **kwargs):
        """
        Match MaskedSoftmax pattern and collect ops to disable output quantizers
        """
        return []
