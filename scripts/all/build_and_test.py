#!/usr/bin/env python3

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Build and test script for AIMET (AI Model Efficiency Toolkit).

This script provides a task-based build system with dependency management.

Usage:
    python scripts/all/build_and_test.py --task <task_name> [options]

Examples:
    # Build AIMET Torch (CPU)
    python scripts/all/build_and_test.py --task build --package aimet_torch

    # Build AIMET Torch (GPU/CUDA)
    python scripts/all/build_and_test.py --task build --package aimet_torch --gpu

    # Build AIMET ONNX (CPU)
    python scripts/all/build_and_test.py --task build --package aimet_onnx

    # Build documentation
    python scripts/all/build_and_test.py --task build --package docs

    # Run tests (skips build if already installed)
    python scripts/all/build_and_test.py --task test --package aimet_torch

    # Run tests with CUDA
    python scripts/all/build_and_test.py --task test --package aimet_torch --gpu

    # Force rebuild even if already installed
    python scripts/all/build_and_test.py --task build --package aimet_torch --force-rebuild

    # List available tasks
    python scripts/all/build_and_test.py --task list
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from util import (
    are_ubuntu_deps_installed,
    get_repo_root,
    get_torch_index_url,
    is_aimet_onnx_installed,
    is_aimet_torch_installed,
    on_linux,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_VENV_PATH = REPO_ROOT / "aimet-dev"
DEFAULT_PYTHON = "python3"

# Valid package choices
VALID_PACKAGES = ["aimet_torch", "aimet_onnx", "docs"]

# Global registries for tasks and dependencies
ALL_TASKS: dict[str, str] = {}
PUBLIC_TASKS: dict[str, str] = {}
TASK_DEPENDENCIES: dict[str, list[str]] = {}


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a task execution."""

    status: TaskStatus
    message: str = ""
    duration_seconds: float = 0.0


class Task(ABC):
    """Base class for all tasks."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    @abstractmethod
    def run(self) -> TaskResult:
        """Execute the task."""
        pass

    def does_work(self) -> bool:
        """Return True if this task does actual work (vs being a no-op)."""
        return True


class BashScriptTask(Task):
    """Task that runs a bash script."""

    def __init__(
        self,
        name: str,
        script_path: str | Path,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        super().__init__(name)
        self.script_path = Path(script_path)
        self.args = args or []
        self.env = env
        self.cwd = cwd or REPO_ROOT

    def run(self) -> TaskResult:
        cmd = [str(self.script_path)] + self.args
        logger.info(f"Running: {' '.join(cmd)}")

        env = os.environ.copy()
        if self.env:
            env.update(self.env)

        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
            return TaskResult(TaskStatus.SUCCESS)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"Script failed with exit code {e.returncode}"
            )


class CommandTask(Task):
    """Task that runs a shell command."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        super().__init__(name)
        self.cmd = cmd
        self.env = env
        self.cwd = cwd or REPO_ROOT

    def run(self) -> TaskResult:
        logger.info(f"Running: {' '.join(self.cmd)}")

        env = os.environ.copy()
        if self.env:
            env.update(self.env)

        try:
            subprocess.run(
                self.cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
            return TaskResult(TaskStatus.SUCCESS)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"Command failed with exit code {e.returncode}"
            )
        except FileNotFoundError as e:
            return TaskResult(TaskStatus.FAILED, f"Command not found: {e}")


class WheelBuildTask(Task):
    """Task that builds a wheel using python -m build (Linux manylinux workflow)."""

    def __init__(
        self,
        name: str,
        cwd: Path | None = None,
        skip_if_installed: Callable[[], bool] | None = None,
        force_rebuild: bool = False,
        enable_torch: bool = False,
        enable_onnx: bool = False,
        enable_cuda: bool = False,
        enable_docs: bool = False,
        editable: bool = False,
    ) -> None:
        super().__init__(name)
        self.cwd = cwd or REPO_ROOT
        self.skip_if_installed = skip_if_installed
        self.force_rebuild = force_rebuild
        self.enable_torch = enable_torch
        self.enable_onnx = enable_onnx
        self.enable_cuda = enable_cuda
        self.enable_docs = enable_docs
        self.editable = editable

        # Build cmake_args from flags
        cuda = "ON" if enable_cuda else "OFF"
        torch = "ON" if enable_torch else "OFF"
        onnx = "ON" if enable_onnx else "OFF"
        self.cmake_args = (
            f"-DENABLE_CUDA={cuda} -DENABLE_TORCH={torch} -DENABLE_ONNX={onnx}"
        )

        # Set skbuild_targets based on enable_docs
        self.skbuild_targets = "all;doc" if enable_docs else "all"

    def _get_manylinux_exclude_libs(self) -> list[str]:
        """Get list of libraries to exclude from manylinux auditwheel repair."""
        exclude_libs: list[str] = []

        # Exclude Torch libraries
        if self.enable_torch:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import torch; print(f'{torch.utils.cmake_prefix_path}/../../lib')",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                torch_dir = result.stdout.strip()
                if torch_dir and Path(torch_dir).exists():
                    # Find all .so files and get their sonames
                    for so_file in Path(torch_dir).glob("*.so*"):
                        try:
                            soname_result = subprocess.run(
                                ["patchelf", "--print-soname", str(so_file)],
                                capture_output=True,
                                text=True,
                            )
                            if (
                                soname_result.returncode == 0
                                and soname_result.stdout.strip()
                            ):
                                exclude_libs.extend(
                                    ["--exclude", soname_result.stdout.strip()]
                                )
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Could not get Torch library excludes: {e}")

        # Exclude CUDA libraries
        if self.enable_cuda:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sysconfig; from pathlib import Path; print(Path(sysconfig.get_config_var('prefix'), 'lib'))",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                cuda_dir = result.stdout.strip()
                if cuda_dir and Path(cuda_dir).exists():
                    for pattern in ["libcu*.so*", "libnv*.so*", "libnp*.so*"]:
                        for so_file in Path(cuda_dir).glob(pattern):
                            try:
                                soname_result = subprocess.run(
                                    ["patchelf", "--print-soname", str(so_file)],
                                    capture_output=True,
                                    text=True,
                                )
                                if (
                                    soname_result.returncode == 0
                                    and soname_result.stdout.strip()
                                ):
                                    exclude_libs.extend(
                                        ["--exclude", soname_result.stdout.strip()]
                                    )
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Could not get CUDA library excludes: {e}")

        # Exclude libpython
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sysconfig; from pathlib import Path; print(Path(sysconfig.get_config_var('prefix'), 'lib'))",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            libs_dir = result.stdout.strip()
            if libs_dir and Path(libs_dir).exists():
                for so_file in Path(libs_dir).glob("libpython*.so*"):
                    try:
                        soname_result = subprocess.run(
                            ["patchelf", "--print-soname", str(so_file)],
                            capture_output=True,
                            text=True,
                        )
                        if (
                            soname_result.returncode == 0
                            and soname_result.stdout.strip()
                        ):
                            exclude_libs.extend(
                                ["--exclude", soname_result.stdout.strip()]
                            )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Could not get libpython excludes: {e}")

        return exclude_libs

    def run(self) -> TaskResult:
        # Check if we can skip the build
        if (
            not self.force_rebuild
            and self.skip_if_installed
            and self.skip_if_installed()
        ):
            logger.info(
                "Package already installed, skipping build (use --force-rebuild to override)"
            )
            return TaskResult(TaskStatus.SUCCESS, "Skipped - already installed")

        # Install torch with correct variant (CPU or CUDA) - required before build
        if self.enable_torch:
            torch_index_url = get_torch_index_url(self.enable_cuda)
            logger.info(f"Installing torch from {torch_index_url}")
            torch_install_cmd = [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--index-url",
                torch_index_url,
                "torch",
            ]
            logger.info(f"Running: {' '.join(torch_install_cmd)}")
            try:
                subprocess.run(
                    torch_install_cmd, cwd=self.cwd, check=True, capture_output=False
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to pip
                torch_install_cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    torch_index_url,
                    "torch",
                ]
                logger.info(f"Running: {' '.join(torch_install_cmd)}")
                subprocess.run(
                    torch_install_cmd, cwd=self.cwd, check=True, capture_output=False
                )

        # Compile and install dev dependencies
        dev_requirements_file = Path("/tmp") / "dev_requirements.txt"
        logger.info("Compiling dev dependencies from pyproject.toml")
        compile_cmd = [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "compile",
            "--output-file",
            str(dev_requirements_file),
            "--extra",
            "dev",
            "pyproject.toml",
        ]
        logger.info(f"Running: {' '.join(compile_cmd)}")
        try:
            subprocess.run(compile_cmd, cwd=self.cwd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"pip compile failed with exit code {e.returncode}"
            )
        except FileNotFoundError:
            logger.warning("uv not found, trying pip-compile")
            compile_cmd = [
                sys.executable,
                "-m",
                "piptools",
                "compile",
                "--output-file",
                str(dev_requirements_file),
                "--extra",
                "dev",
                "pyproject.toml",
            ]
            try:
                subprocess.run(
                    compile_cmd, cwd=self.cwd, check=True, capture_output=False
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(
                    f"pip-compile also failed: {e}, skipping dev deps compilation"
                )

        # Install dev dependencies
        if dev_requirements_file.exists():
            logger.info("Installing dev dependencies")
            torch_index_url = get_torch_index_url(self.enable_cuda)
            install_cmd = [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--index-strategy",
                "unsafe-best-match",
                "--extra-index-url",
                torch_index_url,
                "-r",
                str(dev_requirements_file),
            ]
            logger.info(f"Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(
                    install_cmd, cwd=self.cwd, check=True, capture_output=False
                )
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"dev deps install failed with exit code {e.returncode}",
                )
            except FileNotFoundError:
                logger.warning("uv not found, trying pip")
                install_cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--extra-index-url",
                    torch_index_url,
                    "-r",
                    str(dev_requirements_file),
                ]
                try:
                    subprocess.run(
                        install_cmd, cwd=self.cwd, check=True, capture_output=False
                    )
                except subprocess.CalledProcessError as e:
                    return TaskResult(
                        TaskStatus.FAILED,
                        f"pip install failed with exit code {e.returncode}",
                    )

        # Clean previous build artifacts
        for d in [self.cwd / "build", self.cwd / "dist", self.cwd / "wheelhouse"]:
            if d.exists():
                logger.info(f"Removing {d}")
                shutil.rmtree(d)

        env = os.environ.copy()
        env["CMAKE_ARGS"] = self.cmake_args
        env["SKBUILD_BUILD_TARGETS"] = self.skbuild_targets

        # Editable install for local development
        # Note: uv pip doesn't handle editable installs with dynamic package names well,
        # so we use regular pip for editable installs.
        if self.editable:
            # Ensure pip is installed in the venv
            pip_install_cmd = [sys.executable, "-m", "uv", "pip", "install", "pip"]
            logger.info(f"Ensuring pip is installed: {' '.join(pip_install_cmd)}")
            subprocess.run(pip_install_cmd, cwd=self.cwd, capture_output=True)

            # Uninstall any existing aimet packages to avoid conflicts with editable install
            for pkg in ["aimet-torch", "aimet-onnx", "aimet-onnx-torch"]:
                uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", "-y", pkg]
                logger.info(f"Uninstalling existing package: {' '.join(uninstall_cmd)}")
                subprocess.run(uninstall_cmd, cwd=self.cwd, capture_output=True)

            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "-e",
                ".[dev,test]",
            ]
            logger.info(f"Running: CMAKE_ARGS='{self.cmake_args}' {' '.join(cmd)}")
            try:
                subprocess.run(
                    cmd,
                    env=env,
                    cwd=self.cwd,
                    check=True,
                    capture_output=False,
                )
                return TaskResult(TaskStatus.SUCCESS)
            except subprocess.CalledProcessError as e:
                return TaskResult(
                    TaskStatus.FAILED,
                    f"Editable install failed with exit code {e.returncode}",
                )

        # Build wheel
        cmd = [sys.executable, "-m", "build", "--wheel", "--no-isolation", "."]

        logger.info(f"Running: CMAKE_ARGS='{self.cmake_args}' {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"Build failed with exit code {e.returncode}"
            )

        # Run auditwheel repair for manylinux compatibility (Linux only)
        dist_dir = self.cwd / "dist"
        wheelhouse_dir = self.cwd / "wheelhouse"
        wheelhouse_dir.mkdir(exist_ok=True)

        wheel_files = list(dist_dir.glob("aimet*.whl"))
        if wheel_files:
            if on_linux():
                exclude_libs = self._get_manylinux_exclude_libs()
                for wheel_file in wheel_files:
                    auditwheel_cmd = [
                        "auditwheel",
                        "repair",
                        "--plat",
                        "manylinux_2_34_x86_64",
                        *exclude_libs,
                        "-w",
                        str(wheelhouse_dir),
                        str(wheel_file),
                    ]
                    logger.info(f"Running: {' '.join(auditwheel_cmd)}")
                    try:
                        subprocess.run(auditwheel_cmd, check=True, capture_output=False)
                    except subprocess.CalledProcessError as e:
                        return TaskResult(
                            TaskStatus.FAILED,
                            f"auditwheel repair failed with exit code {e.returncode}",
                        )
                    except FileNotFoundError:
                        logger.warning(
                            "auditwheel not found, skipping manylinux repair"
                        )
            else:
                # On non-Linux (macOS, Windows), just copy wheels to wheelhouse
                logger.info(
                    "Skipping auditwheel (not on Linux), copying wheels to wheelhouse"
                )
                for wheel_file in wheel_files:
                    shutil.copy(wheel_file, wheelhouse_dir)

        # Re-tag torch wheels
        if self.enable_torch:
            repaired_wheels = list(wheelhouse_dir.glob("aimet*.whl"))
            for wheel_file in repaired_wheels:
                retag_cmd = [
                    sys.executable,
                    "-m",
                    "wheel",
                    "tags",
                    "--remove",
                    "--python-tag=py310",
                    "--abi-tag=none",
                    "--platform-tag=any",
                    str(wheel_file),
                ]
                logger.info(f"Running: {' '.join(retag_cmd)}")
                try:
                    subprocess.run(retag_cmd, check=True, capture_output=False)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"wheel tags failed: {e}")
                except FileNotFoundError:
                    logger.warning("wheel module not found, skipping re-tagging")

        return TaskResult(TaskStatus.SUCCESS)


class PyTestTask(Task):
    """Task that installs wheel with [test] extras and runs pytest."""

    def __init__(
        self,
        name: str,
        test_paths: list[str | Path],
        markers: str | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        skip_install_if: Callable[[], bool] | None = None,
        parallel: int = 8,
    ) -> None:
        super().__init__(name)
        self.test_paths = [str(p) for p in test_paths]
        self.markers = markers
        self.extra_args = extra_args or []
        self.env = env
        self.cwd = cwd or REPO_ROOT
        self.skip_install_if = skip_install_if
        self.parallel = parallel

    def run(self) -> TaskResult:
        # Check if we can skip wheel installation
        if self.skip_install_if and self.skip_install_if():
            logger.info("Package already installed, skipping wheel installation")
        else:
            # Find the wheel in wheelhouse/
            wheelhouse_dir = self.cwd / "wheelhouse"
            wheel_files = list(wheelhouse_dir.glob("aimet*.whl"))

            if not wheel_files:
                # Fallback to dist/ if wheelhouse is empty
                dist_dir = self.cwd / "dist"
                wheel_files = list(dist_dir.glob("aimet*.whl"))

            if not wheel_files:
                return TaskResult(
                    TaskStatus.FAILED, "No wheel file found in wheelhouse/ or dist/"
                )

            wheel_path = wheel_files[0]
            logger.info(f"Found wheel: {wheel_path}")

            # Install wheel with [test] extras
            install_cmd = [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                f"{wheel_path}[test]",
            ]
            logger.info(f"Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(
                    install_cmd, cwd=self.cwd, check=True, capture_output=False
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to pip
                install_cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    f"{wheel_path}[test]",
                ]
                logger.info(f"Running: {' '.join(install_cmd)}")
                try:
                    subprocess.run(
                        install_cmd, cwd=self.cwd, check=True, capture_output=False
                    )
                except subprocess.CalledProcessError as e:
                    return TaskResult(
                        TaskStatus.FAILED,
                        f"pip install failed with exit code {e.returncode}",
                    )

        # Run pytest
        cmd = [sys.executable, "-m", "pytest"]

        if self.parallel > 0:
            cmd.extend(["-n", str(self.parallel)])

        if self.markers:
            cmd.extend(["-m", self.markers])

        cmd.extend(self.extra_args)
        cmd.extend(self.test_paths)

        logger.info(f"Running: {' '.join(cmd)}")

        env = os.environ.copy()
        if self.env:
            env.update(self.env)

        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=self.cwd,
                check=True,
                capture_output=False,
            )
            return TaskResult(TaskStatus.SUCCESS)
        except subprocess.CalledProcessError as e:
            return TaskResult(
                TaskStatus.FAILED, f"pytest failed with exit code {e.returncode}"
            )


class ListTasksTask(Task):
    """Task that prints available tasks."""

    def __init__(self, tasks: dict[str, str]) -> None:
        super().__init__("list_tasks")
        self.tasks = tasks

    def run(self) -> TaskResult:
        print("\n" + "=" * 80)
        print("AIMET Build and Test Script")
        print("=" * 80)

        print("\nTasks:")
        print("-" * 80)
        for task_name, description in sorted(self.tasks.items()):
            print(f"  {task_name:<15} {description}")
        print("-" * 80)

        print("\nCommon Options:")
        print(
            "  --package PKG     Package to build/test: aimet_torch, aimet_onnx, docs"
        )
        print("  --gpu             Enable GPU/CUDA support (default: CPU only)")
        print("  --skip TASK       Skip specified task(s)")
        print("  --only            Run only the specified task, skip dependencies")
        print("  --dry-run         Print execution plan without running")

        print("\nBuild Options:")
        print("  --editable, -e    Install in editable mode for local development")
        print("  --force-rebuild   Force rebuild even if package is already installed")

        print("\nTest Options:")
        print("  --parallel N, -n  Run tests with N parallel workers (default: 8)")
        print("  --force-rebuild   Force wheel reinstall before testing")

        print("\nExamples:")
        print("  # Build aimet_torch (CPU, wheel)")
        print(
            "  python scripts/all/build_and_test.py --task build --package aimet_torch"
        )
        print()
        print("  # Build aimet_torch (GPU, editable for local dev)")
        print(
            "  python scripts/all/build_and_test.py --task build --package aimet_torch --gpu -e"
        )
        print()
        print("  # Run tests (skips build if already installed)")
        print("  python scripts/all/build_and_test.py --task test --package aimet_onnx")
        print()
        print("  # Build and test with forced rebuild")
        print(
            "  python scripts/all/build_and_test.py --task test --package aimet_torch --force-rebuild"
        )
        print()
        print("  # Run tests only (skip build dependency)")
        print(
            "  python scripts/all/build_and_test.py --task test --package aimet_torch --only"
        )
        print()
        print("=" * 80)

        return TaskResult(TaskStatus.SUCCESS)


def task(func: Callable) -> Callable:
    """Decorator to register a task."""
    task_name = func.__name__
    ALL_TASKS[task_name] = func.__doc__ or ""
    return func


def public_task(description: str) -> Callable:
    """Decorator to register a public (user-facing) task."""

    def decorator(func: Callable) -> Callable:
        task_name = func.__name__
        ALL_TASKS[task_name] = description
        PUBLIC_TASKS[task_name] = description
        return func

    return decorator


def depends(dependencies: list[str]) -> Callable:
    """Decorator to specify task dependencies."""

    def decorator(func: Callable) -> Callable:
        task_name = func.__name__
        TASK_DEPENDENCIES[task_name] = dependencies
        return func

    return decorator


@dataclass
class Plan:
    """Execution plan for tasks."""

    steps: list[str] = field(default_factory=list)
    tasks: dict[str, Task] = field(default_factory=dict)
    results: dict[str, TaskResult] = field(default_factory=dict)

    def add_step(self, task: Task, step_id: str) -> str:
        """Add a step to the plan."""
        if step_id not in self.steps:
            self.steps.append(step_id)
            self.tasks[step_id] = task
        return step_id

    def has_step(self, step_id: str) -> bool:
        """Check if a step exists in the plan."""
        return step_id in self.steps

    def run(self) -> bool:
        """Execute all steps in the plan."""
        import time

        all_success = True
        for step_id in self.steps:
            task = self.tasks[step_id]
            if not task.does_work():
                self.results[step_id] = TaskResult(TaskStatus.SKIPPED, "No-op task")
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Starting task: {step_id}")
            logger.info(f"{'=' * 60}")

            start_time = time.time()
            result = task.run()
            result.duration_seconds = time.time() - start_time

            self.results[step_id] = result

            if result.status == TaskStatus.FAILED:
                logger.error(f"Task {step_id} FAILED: {result.message}")
                all_success = False
                break
            else:
                logger.info(
                    f"Task {step_id} completed in {result.duration_seconds:.2f}s"
                )

        return all_success

    def print(self) -> None:
        """Print the plan (dry-run mode)."""
        print("\nExecution plan:")
        print("-" * 60)
        for i, step_id in enumerate(self.steps, 1):
            task = self.tasks[step_id]
            work_indicator = "" if task.does_work() else " (no-op)"
            print(f"  {i}. {step_id}{work_indicator}")
        print("-" * 60)

    def print_report(self) -> None:
        """Print execution report."""
        print("\n" + "=" * 60)
        print("Execution Report")
        print("=" * 60)

        for step_id, result in self.results.items():
            status_str = result.status.value.upper()
            duration_str = (
                f"{result.duration_seconds:.2f}s" if result.duration_seconds > 0 else ""
            )
            print(f"  {step_id:<40} {status_str:<10} {duration_str}")

        print("=" * 60)


@dataclass
class BuildConfig:
    """Configuration for build tasks."""

    package: str | None = None
    gpu: bool = False
    force_rebuild: bool = False
    editable: bool = False
    parallel: int = 8


class TaskLibrary:
    """Library of all available tasks for AIMET."""

    def __init__(
        self,
        python_executable: str = DEFAULT_PYTHON,
        venv_path: Path | None = None,
        build_dir: Path | None = None,
        build_config: BuildConfig | None = None,
    ) -> None:
        self.python_executable = python_executable
        self.venv_path = venv_path or DEFAULT_VENV_PATH
        self.build_dir = build_dir or DEFAULT_BUILD_DIR
        self.build_config = build_config or BuildConfig()

    # -------------------------------------------------------------------------
    # Dependency Installation Tasks
    # -------------------------------------------------------------------------

    @public_task("Install Ubuntu system dependencies (Linux only)")
    def install_deps(self, plan: Plan) -> str:
        class ConditionalInstallDepsTask(Task):
            def __init__(self) -> None:
                super().__init__("Installing Ubuntu dependencies")

            def run(self) -> TaskResult:
                if not on_linux():
                    logger.info("Skipping Ubuntu deps (not on Linux)")
                    return TaskResult(TaskStatus.SUCCESS, "Skipped - not on Linux")

                if are_ubuntu_deps_installed():
                    logger.info("Ubuntu dependencies already installed, skipping")
                    return TaskResult(TaskStatus.SUCCESS, "Skipped - already installed")

                cmd = [str(REPO_ROOT / "scripts/posix/install_ubuntu_deps.sh")]
                logger.info(f"Running: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=False)
                    return TaskResult(TaskStatus.SUCCESS)
                except subprocess.CalledProcessError as e:
                    return TaskResult(
                        TaskStatus.FAILED,
                        f"Script failed with exit code {e.returncode}",
                    )

        return plan.add_step(ConditionalInstallDepsTask(), "install_deps")

    # -------------------------------------------------------------------------
    # Build Task (unified)
    # -------------------------------------------------------------------------

    @public_task("Build AIMET (use --package and --gpu)")
    @depends(["install_deps"])
    def build(self, plan: Plan) -> str:
        package = self.build_config.package
        gpu = self.build_config.gpu
        force_rebuild = self.build_config.force_rebuild
        editable = self.build_config.editable

        if not package:
            logger.error("--package is required for build task")
            sys.exit(1)

        # Determine build flags based on package
        if package == "docs":
            enable_torch = True
            enable_onnx = True
            enable_docs = True
            skip_check = None
        elif package == "aimet_torch":
            enable_torch = True
            enable_onnx = False
            enable_docs = False
            skip_check = is_aimet_torch_installed
        elif package == "aimet_onnx":
            enable_torch = False
            enable_onnx = True
            enable_docs = False
            skip_check = is_aimet_onnx_installed
        else:
            logger.error(f"Unknown package: {package}")
            sys.exit(1)

        variant = "GPU" if gpu else "CPU"
        mode = "editable" if editable else "wheel"
        return plan.add_step(
            WheelBuildTask(
                f"Building {package} ({variant}, {mode})",
                skip_if_installed=skip_check,
                force_rebuild=force_rebuild,
                enable_torch=enable_torch,
                enable_onnx=enable_onnx,
                enable_cuda=gpu,
                enable_docs=enable_docs,
                editable=editable,
            ),
            "build",
        )

    # -------------------------------------------------------------------------
    # Test Task (unified)
    # -------------------------------------------------------------------------

    @public_task("Run tests (use --package and --gpu)")
    @depends(["build"])
    def test(self, plan: Plan) -> str:
        package = self.build_config.package
        gpu = self.build_config.gpu
        force_rebuild = self.build_config.force_rebuild
        editable = self.build_config.editable

        if not package:
            logger.error("--package is required for test task")
            sys.exit(1)

        if package == "docs":
            logger.error("--package docs does not support tests")
            sys.exit(1)

        if package == "aimet_torch":
            test_paths = [REPO_ROOT / "TrainingExtensions/torch/test/python"]
            markers = None if gpu else "not cuda"
            skip_check = is_aimet_torch_installed
        elif package == "aimet_onnx":
            test_paths = [REPO_ROOT / "TrainingExtensions/onnx/test/python"]
            markers = None if gpu else "not cuda"
            skip_check = is_aimet_onnx_installed
        else:
            logger.error(f"Unknown package: {package}")
            sys.exit(1)

        # Skip wheel installation for editable mode or if already installed (unless --force-rebuild)
        if editable:
            skip_install_if = lambda: True  # Always skip for editable
        elif force_rebuild:
            skip_install_if = None
        else:
            skip_install_if = skip_check

        parallel = self.build_config.parallel
        variant = "GPU" if gpu else "CPU"
        return plan.add_step(
            PyTestTask(
                f"Running {package} tests ({variant})",
                test_paths,
                markers=markers,
                skip_install_if=skip_install_if,
                parallel=parallel,
            ),
            "test",
        )

    # -------------------------------------------------------------------------
    # Utility Tasks
    # -------------------------------------------------------------------------

    @public_task("Clean build artifacts")
    def clean(self, plan: Plan) -> str:
        class CleanTask(Task):
            def __init__(self) -> None:
                super().__init__("Cleaning build artifacts")

            def run(self) -> TaskResult:
                dirs_to_clean = [
                    REPO_ROOT / "build",
                    REPO_ROOT / "dist",
                    REPO_ROOT / "wheelhouse",
                ]
                for d in dirs_to_clean:
                    if d.exists():
                        logger.info(f"Removing {d}")
                        shutil.rmtree(d)

                # Clean egg-info directories
                for egg_info in REPO_ROOT.glob("*.egg-info"):
                    logger.info(f"Removing {egg_info}")
                    shutil.rmtree(egg_info)

                return TaskResult(TaskStatus.SUCCESS)

        return plan.add_step(CleanTask(), "clean")

    @public_task("Create virtual environment using uv")
    def venv(self, plan: Plan) -> str:
        class CreateVenvTask(Task):
            def __init__(self, venv_path: Path) -> None:
                super().__init__("Creating virtual environment")
                self.venv_path = venv_path

            def run(self) -> TaskResult:
                if self.venv_path.exists():
                    logger.info(f"Venv already exists at {self.venv_path}")
                    return TaskResult(TaskStatus.SUCCESS, "Already exists")

                logger.info(f"Creating venv at {self.venv_path}")
                cmd = ["uv", "venv", "--python", "3.10", str(self.venv_path)]
                logger.info(f"Running: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True, capture_output=False)
                except subprocess.CalledProcessError as e:
                    return TaskResult(
                        TaskStatus.FAILED,
                        f"uv venv failed with exit code {e.returncode}",
                    )
                except FileNotFoundError:
                    return TaskResult(
                        TaskStatus.FAILED,
                        "uv not found. Install with: pip install uv",
                    )

                logger.info(
                    f"Venv created. Activate: source {self.venv_path}/bin/activate"
                )
                return TaskResult(TaskStatus.SUCCESS)

        return plan.add_step(CreateVenvTask(self.venv_path), "venv")

    @public_task("Run pre-commit hooks on all files")
    def precommit(self, plan: Plan) -> str:
        return plan.add_step(
            CommandTask(
                "Running pre-commit",
                ["pre-commit", "run", "--all-files"],
            ),
            "precommit",
        )

    # -------------------------------------------------------------------------
    # List Tasks
    # -------------------------------------------------------------------------

    @public_task("List available tasks")
    def list(self, plan: Plan) -> str:
        return plan.add_step(ListTasksTask(PUBLIC_TASKS), "list")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build and test AIMET (AI Model Efficiency Toolkit).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to run: build, test, clean, list, precommit, install_deps",
    )

    parser.add_argument(
        "--package",
        type=str,
        choices=VALID_PACKAGES,
        help="Package to build/test: aimet_torch, aimet_onnx, docs",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU/CUDA support (default: CPU only)",
    )

    parser.add_argument(
        "--skip",
        metavar="TASK",
        type=str,
        nargs="+",
        help="List of tasks to skip.",
    )

    parser.add_argument(
        "--only",
        action="store_true",
        help="Run only the listed task(s), skipping any dependencies.",
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if packages are already installed.",
    )

    parser.add_argument(
        "--editable",
        "-e",
        action="store_true",
        help="Install in editable mode for local development.",
    )

    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=8,
        metavar="N",
        help="Run tests in parallel with N workers (default: 8, requires pytest-xdist).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan, rather than running it.",
    )

    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON,
        help="Python executable path or name.",
    )

    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(DEFAULT_BUILD_DIR),
        help=f"Build directory (default: {DEFAULT_BUILD_DIR}).",
    )

    return parser.parse_args()


def plan_from_dependencies(
    main_tasks: list[str],
    task_library: TaskLibrary,
) -> Plan:
    """Build a plan with all dependencies resolved."""
    plan = Plan()
    work_list = list(reversed(main_tasks))

    for task_name in work_list:
        if not hasattr(task_library, task_name):
            logger.fatal(f"Task '{task_name}' does not exist.")
            sys.exit(1)

    while work_list:
        task_name = work_list.pop()
        if plan.has_step(task_name):
            continue

        unfulfilled_deps: list[str] = []
        for dep in TASK_DEPENDENCIES.get(task_name, []):
            if not plan.has_step(dep):
                unfulfilled_deps.append(dep)
                if not hasattr(task_library, dep):
                    logger.fatal(
                        f"Non-existent task '{dep}' was declared as a dependency for '{task_name}'."
                    )
                    sys.exit(1)

        if not unfulfilled_deps:
            task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
            added_step = task_adder(plan)
            if added_step != task_name:
                logger.warning(
                    f"Task function '{task_name}' added a task with incorrect id '{added_step}'."
                )
        else:
            work_list.append(task_name)
            work_list.extend(reversed(unfulfilled_deps))

    return plan


def plan_from_task_list(
    tasks: list[str],
    task_library: TaskLibrary,
) -> Plan:
    """Build a plan with only the specified tasks (no dependencies)."""
    plan = Plan()
    for task_name in tasks:
        if not hasattr(task_library, task_name):
            logger.fatal(f"Task '{task_name}' does not exist.")
            sys.exit(1)
        task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
        task_adder(plan)
    return plan


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Create build config from args
    build_config = BuildConfig(
        package=args.package,
        gpu=args.gpu,
        force_rebuild=args.force_rebuild,
        editable=args.editable,
        parallel=args.parallel,
    )

    task_library = TaskLibrary(
        python_executable=args.python,
        build_dir=Path(args.build_dir),
        build_config=build_config,
    )

    # Build the plan
    tasks = [args.task]
    if args.only:
        plan = plan_from_task_list(tasks, task_library)
    else:
        plan = plan_from_dependencies(tasks, task_library)

    # Skip specified tasks
    if args.skip:
        plan.steps = [s for s in plan.steps if s not in args.skip]

    # Execute or print the plan
    if args.dry_run:
        plan.print()
        return 0

    try:
        success = plan.run()
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        success = False
    finally:
        print()
        plan.print_report()
        print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
