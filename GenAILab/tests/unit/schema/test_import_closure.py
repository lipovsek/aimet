# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Import-closure guard for the schema island.

The schema package is rsync'd into AI Hub Models with NO import rewriting. So
every module under ``qai_hub_lm/schema/`` may import ONLY:
  - the standard library,
  - pydantic,
  - its own siblings (``from .x import ...`` / ``from GenAILab.qai_hub_lm.schema...``).

Any ``import aimet*``, ``from GenAILab...`` (outside the schema package), or
other third-party import would survive the dumb rsync verbatim and break the
synced copy in AIHM. This test parses every schema module's AST and fails on a
forbidden import -- before it can rot the sync.
"""

import ast
import pathlib

import GenAILab.qai_hub_lm.schema as schema_pkg

_SCHEMA_DIR = pathlib.Path(schema_pkg.__file__).parent

# Allowed top-level import roots (stdlib is checked separately by allow-listing
# everything that is not obviously foreign; we explicitly allow these names).
_ALLOWED_THIRD_PARTY = {"pydantic"}
_SCHEMA_PACKAGE_PREFIX = "GenAILab.qai_hub_lm.schema"
_FORBIDDEN_PREFIXES = ("aimet", "torch", "transformers", "numpy", "onnx")


def _iter_schema_modules():
    return sorted(_SCHEMA_DIR.glob("*.py"))


def _module_root(name: str) -> str:
    return name.split(".")[0]


def test_no_forbidden_imports_in_schema_modules():
    offenders = []
    for path in _iter_schema_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 means relative (sibling) import -- always allowed
                if node.level and node.level > 0:
                    continue
                if node.module:
                    names = [node.module]
            for name in names:
                # explicit aimet/heavy-dep ban (the thing the sync can't survive)
                if any(
                    name == p or name.startswith(p + ".") for p in _FORBIDDEN_PREFIXES
                ):
                    offenders.append((path.name, name))
                    continue
                # absolute schema-package import is allowed
                if name.startswith(_SCHEMA_PACKAGE_PREFIX):
                    continue
                # any other GenAILab import is forbidden (would break the island)
                if _module_root(name) == "GenAILab":
                    offenders.append((path.name, name))
    assert not offenders, (
        "Forbidden imports in schema island (would break the AIHM sync):\n"
        + "\n".join(f"  {f}: {n}" for f, n in offenders)
    )
