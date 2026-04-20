#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Check pinned dependencies against PyPI and write updated pins if newer versions exist."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def get_latest_version(package: str) -> str:
    """Query PyPI for the latest version of a package."""
    result = subprocess.run(
        ["pip", "index", "versions", package],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"Warning: pip index versions failed for {package}: {result.stderr}",
            file=sys.stderr,
        )
        return ""

    # Output format: "package (X.Y.Z)" on the first line
    match = re.search(r"\(([^)]+)\)", result.stdout.split("\n")[0])
    if not match:
        print(f"Warning: could not parse version for {package}", file=sys.stderr)
        return ""
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pinned_file", type=Path, help="Path to requirements-pinned.txt"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to write updated pins"
    )
    args = parser.parse_args()

    lines = args.pinned_file.read_text().splitlines()
    updated = []
    has_changes = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated.append(line)
            continue

        match = re.match(r"^([a-zA-Z0-9_-]+)==(.+)$", stripped)
        if not match:
            updated.append(line)
            continue

        pkg, current = match.group(1), match.group(2)
        latest = get_latest_version(pkg)

        if not latest:
            # Could not determine latest version; keep current pin
            updated.append(line)
            continue

        if latest != current:
            print(f"{pkg}: {current} -> {latest}")
            has_changes = True

        updated.append(f"{pkg}=={latest}")

    args.output.write_text("\n".join(updated) + "\n")

    if has_changes:
        print("Updates found.")
    else:
        print("All packages are up to date.")


if __name__ == "__main__":
    main()
