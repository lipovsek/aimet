# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate a compact markdown summary table for the workflow results job.

Reads per-framework comparison_summary.json artifacts and produces a
markdown table showing test counts, baseline comparison, quantization
status, and any issues.

Usage:
    python -m AIMETRegression.workflow.summary \
        --onnx-status success \
        --torch-status success \
        --onnx-summary summaries/onnx/comparison_summary.json \
        --torch-summary summaries/torch/comparison_summary.json \
        --suite nightly \
        --trigger schedule
"""

import argparse
import json
import sys


def load_summary(path):
    """Load a comparison summary JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status_emoji(status):
    """Map job status to display string."""
    return {"success": "✅ Passed", "skipped": "⏭️ Skipped"}.get(status, "❌ Failed")


def branch_prefix(branch):
    """Title-prefix that distinguishes develop runs from release-branch runs."""
    if not branch:
        return ""
    if branch.startswith("release-"):
        return f"[{branch}] 🚨 "
    return f"[{branch}] "


def baseline_cell(data):
    """Format baseline comparison column."""
    parts = []
    if data["stable"]:
        parts.append(f"✅ Stable: {data['stable']}")
    if data["improvements"]:
        parts.append(f"📈 Improvements: {data['improvements']}")
    if data["regressions"]:
        parts.append(f"⚠️ Regressions: {data['regressions']}")
    return "<br>".join(parts) if parts else "N/A"


def quality_cell(data):
    """Format quantization status column."""
    parts = []
    if data["passed"]:
        parts.append(f"✅ Passed: {data['passed']}")
    if data["warnings"]:
        parts.append(f"⚠️ Warnings: {data['warnings']}")
    if data["failed"]:
        parts.append(f"❌ Failed: {data['failed']}")
    if data.get("crashed"):
        parts.append(f"💥 Crashed: {data['crashed']}")
    return "<br>".join(parts) if parts else "N/A"


def generate_summary(
    onnx_status, torch_status, onnx_summary_path, torch_summary_path, suite, trigger
):
    """Generate and print the markdown summary to stdout."""
    onnx = load_summary(onnx_summary_path) if onnx_summary_path else None
    torch_data = load_summary(torch_summary_path) if torch_summary_path else None

    print(f"## Nightly Regression Results\n")
    print(f"**Suite:** {suite} | **Trigger:** {trigger}\n")

    # Results table
    print("| Framework | Status | Tests | Baseline Comparison | Quantization Status |")
    print("|-----------|--------|-------|---------------------|---------------------|")

    for name, status, data in [
        ("ONNX", onnx_status, onnx),
        ("Torch", torch_status, torch_data),
    ]:
        emoji = status_emoji(status)
        if data:
            tests = data["total"]
            baseline = baseline_cell(data)
            quality = quality_cell(data)
        elif status == "skipped":
            tests = "-"
            baseline = "-"
            quality = "-"
        else:
            tests = "?"
            baseline = "-"
            quality = "-"
        print(f"| {name} | {emoji} | {tests} | {baseline} | {quality} |")

    # Failed tests detail
    failed_sections = []
    for name, data in [("ONNX", onnx), ("Torch", torch_data)]:
        if not data:
            continue
        items = []
        if data["regression_tests"]:
            items.append(("⚠️ Regressions", data["regression_tests"]))
        if data["failed_tests"]:
            items.append(("❌ Quantization failures", data["failed_tests"]))
        if items:
            failed_sections.append((name, items))

    if failed_sections:
        print("\n### Issues\n")
        for name, items in failed_sections:
            for label, tests in items:
                print(f"**{name}** — {label}:")
                for t in tests:
                    print(f"- `{t}`")
                print()


def generate_slack_summary(
    onnx_status,
    torch_status,
    onnx_summary_path,
    torch_summary_path,
    suite,
    trigger,
    run_url=None,
    branch="",
):
    """Generate a Slack-friendly plain text summary to stdout."""
    print(f"*{branch_prefix(branch)}{suite.title()} Regression failed*")
    if run_url:
        print(f"Workflow: {run_url}")


def main():
    parser = argparse.ArgumentParser(description="Generate workflow results summary")
    parser.add_argument("--onnx-status", default="skipped")
    parser.add_argument("--torch-status", default="skipped")
    parser.add_argument("--onnx-summary", default=None)
    parser.add_argument("--torch-summary", default=None)
    parser.add_argument("--suite", default="nightly")
    parser.add_argument("--trigger", default="unknown")
    parser.add_argument("--format", default="github", choices=["github", "slack"])
    parser.add_argument("--run-url", default=None)
    parser.add_argument(
        "--branch",
        default="",
        help="git ref name, used to distinguish develop vs release-branch runs",
    )
    args = parser.parse_args()

    if args.format == "slack":
        generate_slack_summary(
            onnx_status=args.onnx_status,
            torch_status=args.torch_status,
            onnx_summary_path=args.onnx_summary,
            torch_summary_path=args.torch_summary,
            suite=args.suite,
            trigger=args.trigger,
            run_url=args.run_url,
            branch=args.branch,
        )
    else:
        generate_summary(
            onnx_status=args.onnx_status,
            torch_status=args.torch_status,
            onnx_summary_path=args.onnx_summary,
            torch_summary_path=args.torch_summary,
            suite=args.suite,
            trigger=args.trigger,
        )


if __name__ == "__main__":
    main()
