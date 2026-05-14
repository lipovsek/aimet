#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

set -e

if ! command -v argo &>/dev/null; then
  echo "Installing Argo CLI..."
  curl -fsSL "https://github.com/argoproj/argo-workflows/releases/download/v3.5.5/argo-linux-amd64.gz" | gunzip > /tmp/argo
  sudo install /tmp/argo /usr/local/bin/argo
  rm /tmp/argo
fi

if ! command -v kubectl &>/dev/null; then
  echo "Installing kubectl..."
  curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /tmp/kubectl
  sudo install /tmp/kubectl /usr/local/bin/kubectl
  rm /tmp/kubectl
fi
