#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Script to do the following:
# 1. Obtain the minimum supported value of glibc by invoking a separate script
# 2. Rename the package wheel files to conform to manylinux format for PyPi

# Usage:
# convert_wheel_format.sh <build_path> [<package_wheel_path>]

# verbose mode
# set -x

# enable exit on error
set -e

# Set the build path
if [[ -z ${1} ]]; then
    echo "No Binary directory specified"
    exit 3
fi
BUILD_DIR=$1

# Set the default packaging wheel files path relative to the build path
WHEEL_DIR="${BUILD_DIR}/packaging/dist"
if [[ ${2} ]]; then
    # Override with a custom package path if provided
    WHEEL_DIR=$2
fi

# Search for and create an array of .so files in the package
declare -a so_file_list=($(find ${BUILD_DIR} -name *.so | grep artifacts))

# Loop through each library file
for so_file in "${so_file_list[@]}"
do
    # Find glibc version and append to list
    glibc_ver=$(echo $so_file | xargs -I {} objdump -T {} | grep GLIBC | sed 's/.*GLIBC_\([.0-9]*\).*/\1/g' | sort -Vu | tail -1)
    echo "glibc version for $so_file is $glibc_ver"
    glibc_ver_list+="$(echo "$glibc_ver ")"
done

# Sort the array and determine the smallest value
glibc_ver_list_sorted=$(echo ${glibc_ver_list} | xargs -n1 | sort -r --version-sort | xargs)
glibc_min_ver=$(echo "${glibc_ver_list_sorted}" | awk '{print $1;}' | tr '.' '_')
echo "glibc_min_ver = ${glibc_min_ver}"

# Rename the package wheel files to conform to manylinux format for PyPi
whlfile_name=$(find ${WHEEL_DIR}/*.whl | head -n 1)
if [[ $whlfile_name =~ aimet_torch.+ ]]; then
    # aimet_torch supports all platform by default,
    # and only optionally asserts certain platform when importing aimet_torch.v1
    tags="--abi-tag=none --platform-tag=any"
else
    tags="--platform-tag=manylinux_${glibc_min_ver}_x86_64"
fi
wheel tags ${tags} --remove $whlfile_name
