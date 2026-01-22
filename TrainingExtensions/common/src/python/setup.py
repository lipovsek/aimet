# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import copy

from setuptools import Distribution, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext

CURRENT_DIR = Path(__file__).parent.resolve()
PACKAGING_DIR = CURRENT_DIR / ".." / ".." / ".." / ".." / "packaging"


def str2bool(str_):
    TRUE_VALS = {"true", "yes", "y", "on", "1"}
    FALSE_VALS = {"false", "no", "n", "off", "0"}
    if str_.lower() in TRUE_VALS:
        return True
    elif str_.lower() in FALSE_VALS:
        return False
    else:
        raise RuntimeError(
            f"Unknown boolean value '{str_}' (known values are: {TRUE_VALS | FALSE_VALS}"
        )


if {"AIMET_CU_VER", "AIMET_PT_VER"} > os.environ.keys():
    raise RuntimeError(
        "Please specify what veriosn of cuda and pytorch you would like"
        "to use using environment variables: AIMET_CU_VER, AIMET_PT_VER"
    )

ENABLE_CUDA = str2bool(os.environ.get("ENABLE_CUDA", "False"))
ENABLE_TORCH = str2bool(os.environ.get("ENABLE_TORCH", "True"))

PKG_FILES = [
    "dependencies/reqs_pip_common.txt",
    "dependencies/reqs_deb_common.txt",
    "INSTALL.txt",
    "envsetup.sh",
    # "LICENSE.pdf",
]

AIMET_COMMON_VERSION = os.environ.get("SW_VERSION")
if AIMET_COMMON_VERSION is None:
    AIMET_COMMON_VERSION = (PACKAGING_DIR / "version.txt").read_text().strip()

AIMET_COMMON_URL = (
    subprocess.run(
        shlex.split("git config --get remote.origin.url"),
        check=True,
        cwd=CURRENT_DIR,
        stdout=subprocess.PIPE,
        encoding="utf8",
    ).stdout
    + f"/releases/download/{AIMET_COMMON_VERSION}"
)


class BuildExtensionCommand(build_ext):
    def run(self):
        super().run()
        # Create dest directories
        dst_dir = Path(self.get_ext_fullpath("dummy")).parent / "aimet_common"
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "bin").mkdir(parents=True, exist_ok=True)

        # Check if prebuilt C++ part is available
        whl_prep_dir = os.environ.get("WHL_PREP_DIR")
        if whl_prep_dir is None:
            src_dir = CURRENT_DIR
            bld_dir = Path(self.build_temp).resolve()
            whl_prep_dir = bld_dir / "whlprep"
            tgt = f"whl_prep_{'ln' if self.inplace else 'cp'}_"
            cmake_args = [
                f"-DPython3_ROOT_DIR={os.path.dirname(sys.executable)}",
                f"-DWHL_PREP_DIR={whl_prep_dir}",
                f"-DENABLE_CUDA={'OFF' if os.environ['AIMET_CU_VER'] == 'cpu' else 'ON'}",
                f"-DENABLE_TORCH={'OFF' if os.environ['AIMET_PT_VER'] == '' else 'ON'}",
            ]
            subprocess.run(
                ["cmake", "-B", bld_dir, "-S", src_dir] + cmake_args,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                encoding="utf8",
            )
            subprocess.run(
                ["cmake", "--build", bld_dir, "-j", "-t", tgt + "common"],
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                encoding="utf8",
            )
            if os.environ["AIMET_PT_VER"]:
                subprocess.run(
                    ["cmake", "--build", bld_dir, "-j", "-t", tgt + "torch"],
                    check=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    encoding="utf8",
                )
        # Copy C++ part into wheel package
        subprocess.run(
            shlex.split(f"cp -Prv {whl_prep_dir}/aimet_common/. {dst_dir}"),
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            encoding="utf8",
        )
        if os.environ["AIMET_PT_VER"]:
            subprocess.run(
                shlex.split(f"cp -Prv {whl_prep_dir}/aimet_torch/. {dst_dir}"),
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                encoding="utf8",
            )
        subprocess.run(
            shlex.split(
                f"cp -Lrv {' '.join(str(PACKAGING_DIR / f) for f in PKG_FILES)} {dst_dir / 'bin'}"
            ),
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            encoding="utf8",
        )


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    author_email="aimet.os@quicinc.com",
    author="Qualcomm Innovation Center, Inc.",
    cmdclass={
        "build_ext": BuildExtensionCommand,
    },
    description="AIMET Common Package",
    distclass=BinaryDistribution,
    install_requires=list(
        filter(
            lambda r: not r.startswith("-"),
            subprocess.run(
                [sys.executable, str(PACKAGING_DIR / "dependencies.py"), "pip"],
                check=True,
                stdout=subprocess.PIPE,
                encoding="utf8",
            ).stdout.splitlines(),
        )
    ),
    license="NOTICE.txt",
    long_description=(PACKAGING_DIR / "README.txt").read_text(),
    name="AimetCommon",
    package_data={
        "": ["*.json"],
    },
    package_dir={
        "": ".",
    },
    packages=find_namespace_packages(where=".", exclude=["build", "x86_64-linux-gnu"]),
    platforms="x86",
    python_requires=">=3.6",
    url=AIMET_COMMON_URL,
    version=AIMET_COMMON_VERSION,
)
