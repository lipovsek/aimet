# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for AIMET build scripts."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def on_linux() -> bool:
    """Check if running on Linux."""
    return platform.uname().system == "Linux"


def on_macos() -> bool:
    """Check if running on macOS."""
    return platform.uname().system == "Darwin"


def get_repo_root() -> Path:
    """Get repository root using git."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not find repo root. Are you in a git repository?")
    return Path(result.stdout.strip())


def is_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {package_name}"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_aimet_torch_installed() -> bool:
    """Check if AIMET Torch variant is installed."""
    return is_package_installed("aimet_torch")


def is_aimet_onnx_installed() -> bool:
    """Check if AIMET ONNX variant is installed."""
    return is_package_installed("aimet_onnx")


def get_cuda_version() -> str:
    """Get CUDA version from nvcc."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        for line in output.split("\n"):
            if "release" in line:
                version = line.split("release")[-1].strip().split(",")[0]
                return version.replace(".", "")
    except Exception:
        pass
    return ""


def get_torch_index_url(enable_cuda: bool) -> str:
    """Get PyTorch extra-index-url based on CUDA setting."""
    if enable_cuda:
        cuda_version = get_cuda_version()
        if cuda_version:
            return f"https://download.pytorch.org/whl/cu{cuda_version}"
        return "https://download.pytorch.org/whl/cu124"  # Default CUDA version
    return "https://download.pytorch.org/whl/cpu"


def are_ubuntu_deps_installed() -> bool:
    """Check if required Ubuntu dependencies are already installed."""
    if not on_linux():
        return True  # Skip check on non-Linux
    required_packages = ["cmake", "g++-10", "patchelf", "libeigen3-dev", "pandoc"]
    try:
        result = subprocess.run(
            ["dpkg", "-s"] + required_packages,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False
        # Also check for uv (installed via curl, not apt)
        if not shutil.which("uv"):
            return False
        return True
    except Exception:
        return False


def are_macos_deps_installed() -> bool:
    """Check if required macOS dependencies are already installed."""
    if not on_macos():
        return True  # Skip check on non-macOS
    required_packages = ["cmake", "eigen", "pandoc", "pkg-config", "uv"]
    try:
        result = subprocess.run(
            ["brew", "list"] + required_packages,
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False
