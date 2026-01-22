# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=all
import glob
import importlib.util
import os
import pathlib
import subprocess
import sys
import sysconfig


_PKG_ROOT = (
    pathlib.Path(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    .absolute()
    .resolve()
)


def import_from_path(module_name, file_path):
    # From https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


aimet_packaging_plugin = import_from_path(
    module_name="aimet", file_path=_PKG_ROOT / "./packaging/plugins/local/aimet.py"
)


def _get_min_glibc_version(build_dir):
    if sys.platform != "linux":
        return None

    so_file_list = glob.glob(os.path.join(build_dir, "**/*.so"), recursive=True)
    so_file_list = [file for file in so_file_list if "artifacts" in file]
    glibc_ver_list = []
    try:
        for so_file in so_file_list:
            command = f"objdump -T {so_file} | grep GLIBC | sed 's/.*GLIBC_\\([.0-9]*\\).*/\\1/g' | sort -Vu | tail -1"
            glibc_ver = (
                subprocess.check_output(command, shell=True).decode("utf-8").strip()
            )
            glibc_ver_list.append(glibc_ver)
    except subprocess.CalledProcessError:
        return None

    if not glibc_ver_list:
        return None

    return sorted(
        glibc_ver_list, key=lambda x: list(map(int, x.split("."))), reverse=True
    )[0]


def main(output_dir, build_dir):
    try:
        import torch
    except ImportError:
        torch = None

    min_glibc_version = _get_min_glibc_version(build_dir)

    _template = os.path.join(os.path.dirname(__file__), "_version.pyi")

    with open(_template) as f:
        copyright_string = [
            line.strip() for line in f.readlines() if line.startswith("#")
        ]

    content = [
        f"__version__ = '{aimet_packaging_plugin.get_version()}'",
        f"python_abi = '{sysconfig.get_config_var('SOABI')}'",
        "torch = " + (f"'{torch.__version__}'" if torch else "None"),
        "min_glibc = " + (f"'{min_glibc_version}'" if min_glibc_version else "None"),
        # "onnx = "       + (f"'{onnx.__version__}'" if onnx else "None"),
        "",
    ]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "_version.py"), "w") as f:
        f.write("\n".join(copyright_string + content))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--build-dir", type=str)
    args = parser.parse_args()
    main(args.output_dir, args.build_dir)
