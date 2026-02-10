# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX graph fusion pass registry"""

from typing import Type

from onnxscript.rewriter import pattern

from aimet_onnx.graph_passes.pass_registry import BaseRegistry

# Domain for AIMET supergroup functions
AIMET_SUPERGROUP_DOMAIN = "aimet.supergroup"


class FusionPassRegistry(BaseRegistry[list[pattern.RewriteRule]]):
    """
    Registry for onnxscript RewriteRule passes.
    """

    def register(
        self, pass_cls: pattern.RewriteRule, name: str, override: bool = False
    ):
        """
        Register Graph Pass

        Args:
            pass_cls (pattern.RewriteRule): Rewrite rule being registered.
            name (str): Pass name to register graph pass with.
            override (bool, optional): Override existing passes if set. Otherwise extends the existing rule list.
        """
        if name in self.passes and not override:
            self.passes[name].append(pass_cls)
        else:
            self.passes[name] = [pass_cls]


# Global Pass Registry to hold all onnxscript rewrite rule passes
FUSION_PASS_REGISTRY = FusionPassRegistry()


def register_fusion(name: str, override: bool = False, **kwargs):
    """
    Decorator to register an onnxscript rewrite rule pass.

    Args:
        name: Pass name to register with.
        override: Override pass if already registered. Defaults to False.
    """

    def wrapper(
        pass_cls: Type[pattern.RewriteRuleClassBase],
    ) -> Type[pattern.RewriteRuleClassBase]:
        rule = pass_cls.rule(as_function=True, **kwargs)
        FUSION_PASS_REGISTRY.register(rule, name, override)
        return pass_cls

    return wrapper
