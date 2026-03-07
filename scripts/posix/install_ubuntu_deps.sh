#!/usr/bin/bash

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Only run on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Skipping Ubuntu deps installation (not on Linux)"
    exit 0
fi

REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/all/util/common.sh"

set_strict_mode

run_as_root apt-get update
run_as_root apt-get install -y \
  acl \
  ca-certificates \
  cmake \
  curl \
  g++-10 \
  git \
  jq \
  libeigen3-dev \
  libz-dev \
  make \
  pandoc \
  patchelf \
  pkg-config \
  python3 \
  python3-dev \
  sudo \
  build-essential

run_as_root update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10
