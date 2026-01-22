# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
ImageNet dataset setup for dataloader
"""

dataset = {
    "image_width": 224,
    "image_height": 224,
    "image_channels": 3,
    "image_size": 224,
    "images_mean": [0.485, 0.456, 0.406],
    "images_std": [0.229, 0.224, 0.225],
    "images_classes": 1000,
    "val_images_len": 50000,
    "test_images_len": 1281167,
}

evaluation = {"batch_size": 32, "num_workers": 1}

train = {"batch_size": 16, "num_workers": 1}
