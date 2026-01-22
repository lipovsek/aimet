# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Abstract ModelModule class"""

from abc import ABC

from .utils import ModelApi


class ModelModule(ABC):
    """Abstract ModelModule class to represent any of the following: pytorch module or ONNX node"""

    def __init__(self, model_module):
        self._model_module = model_module

    def get_module(self):
        """Getter for module"""
        return self._model_module


class PytorchModelModule(ModelModule):
    """Pytorch ModelModule class to represent a module inside a Pytorch model"""

    def __init__(self, model_module):
        super().__init__(model_module)
        self._api = ModelApi.pytorch


class ONNXModelModule(ModelModule):
    """Keras ModelModule class to represent an op inside a Keras model"""

    def __init__(self, model_module):
        super().__init__(model_module)
        self._api = ModelApi.onnx
