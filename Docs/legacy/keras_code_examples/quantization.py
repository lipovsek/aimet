# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import tensorflow as tf

from aimet_tensorflow.keras import quantsim
# Optional import only required for fine-tuning
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper

def evaluate(model: tf.keras.Model, forward_pass_callback_args):
    """
    This is intended to be the user-defined model evaluation function. AIMET requires the above signature. So if the
    user's eval function does not match this signature, please create a simple wrapper.
    Use representative dataset that covers diversity in training data to compute optimal encodings.

    :param model: Model to evaluate
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
           the user to determine the type of this parameter. E.g. could be simply an integer representing the number
           of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
           If set to None, forward_pass_callback will be invoked with no parameters.
    """
    dummy_x, _ = forward_pass_callback_args
    model(dummy_x)

def quantize_model():
    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=10)
    sim = quantsim.QuantizationSimModel(model)

    # Generate some dummy data
    dummy_x = np.random.randn(10, 224, 224, 3)
    dummy_y = np.random.randint(0, 10, size=(10,))
    dummy_y = tf.keras.utils.to_categorical(dummy_y, num_classes=10)

    # Compute encodings
    sim.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    sim.compute_encodings(evaluate, forward_pass_callback_args=(dummy_x, dummy_y))

    # Do some fine-tuning
    # Note:: For GPU workloads and models with non-trainable BatchNorms is not supported,
    # So user need to explicitly set the BatchNorms to trainable.
    # Below code snippet sets the BatchNorms to trainable
    for layer in sim.model.layers:
        if isinstance(layer, QcQuantizeWrapper) and isinstance(layer._layer_to_wrap, tf.keras.layers.BatchNormalization):
            layer._layer_to_wrap.trainable = True

    sim.model.fit(x=dummy_x, y=dummy_y, epochs=10)

quantize_model()
