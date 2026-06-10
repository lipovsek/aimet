# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Render a GitHub issue body (Markdown) and title for a regression run failure.

Reads comparison_summary.json (one or both frameworks) plus optional results CSVs,
prints the body to stdout, and writes the title to a separate file via --title-out.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


_TITLE_PREFIX = {
    "nightly-onnx": "[Nightly ONNX]",
    "nightly-torch": "[Nightly Torch]",
    "weekly-onnx": "[Weekly ONNX]",
    "weekly-torch": "[Weekly Torch]",
    "nightly": "[Nightly]",
    "weekly": "[Weekly]",
}


def _branch_prefix(branch: str) -> str:
    """Return a title-prefix that distinguishes develop runs from release-branch runs."""
    if not branch:
        return ""
    if branch.startswith("release-"):
        return f"[{branch}] 🚨 "
    return f"[{branch}] "


def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_csv_errors(csv_path: str | None) -> dict[str, str]:
    """Map 'Model/Feature' -> Error string for crashed rows. Empty if no CSV."""
    errors: dict[str, str] = {}
    if not csv_path:
        return errors
    p = Path(csv_path)
    if not p.exists():
        return errors
    with p.open() as f:
        for row in csv.DictReader(f):
            if (row.get("Status") or "").strip() != "crashed":
                continue
            key = f"{row.get('Model', '')}/{row.get('Feature', '')}"
            errors[key] = (row.get("Error") or "").strip()
    return errors


def _section(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    lines = [f"### {title} ({len(items)})", ""]
    lines.extend(f"- `{item}`" for item in items)
    lines.append("")
    return lines


def _crash_section(items: list[str], errors: dict[str, str]) -> list[str]:
    if not items:
        return []
    lines = [f"### 🚫 Not run ({len(items)})", ""]
    for item in items:
        err = errors.get(item, "")
        if err:
            lines.append(f"- `{item}` — `{err}`")
        else:
            lines.append(f"- `{item}`")
    lines.append("")
    return lines


def render(
    *,
    onnx_summary: dict | None,
    torch_summary: dict | None,
    onnx_errors: dict[str, str],
    torch_errors: dict[str, str],
    run_url: str | None,
    suite: str,
    branch: str = "",
) -> tuple[str, str]:
    """Return (title, body_markdown)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    counts = {"crashed": 0, "failed": 0, "regressions": 0}
    for s in (onnx_summary, torch_summary):
        if not s:
            continue
        counts["crashed"] += s.get("crashed", 0)
        counts["failed"] += s.get("failed", 0)
        counts["regressions"] += s.get("regressions", 0)

    title_parts = []
    if counts["crashed"]:
        title_parts.append(f"{counts['crashed']} not run")
    if counts["regressions"]:
        title_parts.append(f"{counts['regressions']} regressed")
    if counts["failed"]:
        title_parts.append(f"{counts['failed']} accuracy failures")
    summary_str = ", ".join(title_parts) if title_parts else "failed"
    prefix = _TITLE_PREFIX.get(suite, f"[{suite}]")
    title = f"{_branch_prefix(branch)}{prefix} Regression failed {today}: {summary_str}"

    body: list[str] = []
    body.append(f"Regression run failed for **{suite}** on {today}.")
    body.append("")
    body.append("## Summary")
    body.append("")
    body.append(
        "| Framework | Total | Not run | Accuracy failures | Baseline regressions |"
    )
    body.append("|---|---:|---:|---:|---:|")
    for fw_name, s in [("ONNX", onnx_summary), ("Torch", torch_summary)]:
        if not s:
            body.append(f"| {fw_name} | — | — | — | — |")
            continue
        body.append(
            f"| {fw_name} | {s.get('total', 0)} | {s.get('crashed', 0)} | "
            f"{s.get('failed', 0)} | {s.get('regressions', 0)} |"
        )
    body.append("")

    for fw_name, s, errs in [
        ("ONNX", onnx_summary, onnx_errors),
        ("Torch", torch_summary, torch_errors),
    ]:
        if not s:
            continue
        fw_lines: list[str] = []
        fw_lines.extend(_crash_section(s.get("crashed_tests", []), errs))
        fw_lines.extend(
            _section("⚠️ Baseline regressions", s.get("regression_tests", []))
        )
        fw_lines.extend(_section("❌ Accuracy failures", s.get("failed_tests", [])))
        if fw_lines:
            body.append(f"## {fw_name}")
            body.append("")
            body.extend(fw_lines)

    if run_url:
        body.append("## Links")
        body.append("")
        body.append(f"- [Workflow run]({run_url})")
        body.append(f"- [HTML report]({run_url}#artifacts) (download from artifacts)")
        body.append("")

    return title, "\n".join(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-summary", default=None)
    parser.add_argument("--torch-summary", default=None)
    parser.add_argument(
        "--onnx-csv",
        default=None,
        help="results CSV for ONNX, used to inline crash error strings",
    )
    parser.add_argument(
        "--torch-csv",
        default=None,
        help="results CSV for Torch, used to inline crash error strings",
    )
    parser.add_argument("--run-url", default=None)
    parser.add_argument(
        "--suite",
        default="nightly",
        help="suite name (nightly|weekly|nightly-onnx|...)",
    )
    parser.add_argument(
        "--title-out", default=None, help="path to write the issue title"
    )
    parser.add_argument(
        "--branch",
        default="",
        help="git ref name, used to distinguish develop vs release-branch runs",
    )
    args = parser.parse_args()

    title, body = render(
        onnx_summary=_load_json(args.onnx_summary),
        torch_summary=_load_json(args.torch_summary),
        onnx_errors=_load_csv_errors(args.onnx_csv),
        torch_errors=_load_csv_errors(args.torch_csv),
        run_url=args.run_url,
        suite=args.suite,
        branch=args.branch,
    )

    sys.stdout.write(body)
    if not body.endswith("\n"):
        sys.stdout.write("\n")

    if args.title_out:
        Path(args.title_out).write_text(title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
