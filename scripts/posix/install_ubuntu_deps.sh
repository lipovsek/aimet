#!/usr/bin/bash

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

set -eu

# Only run on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Skipping Ubuntu deps installation (not on Linux)"
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

. "${SCRIPT_DIR}/../all/util/common.sh"

set_strict_mode

run_as_root apt-get update
run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  acl \
  ca-certificates \
  clang \
  cmake \
  curl \
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

# Install uv
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add to PATH for current session
  export PATH="$HOME/.local/bin:$PATH"
fi

# Set clang as the default compiler
run_as_root update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100
run_as_root update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100

echo "Ubuntu dependencies installed successfully."
echo "Using clang as the default compiler."
