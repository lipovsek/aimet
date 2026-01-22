# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Data loader for ImageNet"""

import os
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torch_data


class ImageNetDataLoader:
    """ImageNet Data Loader"""

    def __init__(self, images_dir, size, batch_size, num_workers):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(size + 24),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self):
        """Property that exposes a data loader for training samples"""

        root = os.path.join(self.images_dir, "train")
        if not self._train_loader:
            train_set = torchvision.datasets.ImageFolder(
                root=root, transform=self.train_transforms
            )
            self._train_loader = torch_data.DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._train_loader

    @property
    def val_loader(self):
        """Property that exposes a data loader for validation samples"""

        root = os.path.join(self.images_dir, "val")
        if not self._val_loader:
            val_set = torchvision.datasets.ImageFolder(
                root=root, transform=self.val_transforms
            )
            self._val_loader = torch_data.DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._val_loader
