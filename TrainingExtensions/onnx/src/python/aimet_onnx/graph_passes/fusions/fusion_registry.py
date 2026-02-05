# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX graph fusion pass registry"""

from typing import Type

from onnxscript.rewriter import pattern

from aimet_onnx.graph_passes.pass_registry import BaseRegistry

# Domain for AIMET supergroup functions
AIMET_SUPERGROUP_DOMAIN = "aimet.supergroup"


class FusionPassRegistry(BaseRegistry[pattern.RewriteRuleClassBase]):
    """
    Registry for onnxscript RewriteRule passes.
    """


# Global Pass Registry to hold all onnxscript rewrite rule passes
FUSION_PASS_REGISTRY = FusionPassRegistry()


def register_fusion(name: str, override: bool = False):
    """
    Decorator to register an onnxscript rewrite rule pass.

    Args:
        name: Pass name to register with.
        override: Override pass if already registered. Defaults to False.
    """

    def wrapper(
        pass_cls: Type[pattern.RewriteRuleClassBase],
    ) -> Type[pattern.RewriteRuleClassBase]:
        FUSION_PASS_REGISTRY.register(pass_cls, name, override)
        return pass_cls

    return wrapper
