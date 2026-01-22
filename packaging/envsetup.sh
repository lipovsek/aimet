#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# This script sets up the various environment variables needed to use various modules and scripts

#====================================
# Configuration of ENV variables
#===================================
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
if [ -z "${LD_LIBRARY_PATH}" ]; then
LD_LIBRARY_PATH=""
fi
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_VISIBLE_DEVICES=all
