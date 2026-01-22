# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file

""" AdaRound code example to be used for documentation generation. """

# AdaRound imports

import logging
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.adaround_weight import Adaround
from aimet_tensorflow.keras.adaround_weight import AdaroundParameters

# End of import statements

def dummy_forward_pass(model: tf.keras.Model, _):
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.
    :param model: Model to be evaluated
    :param _: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
    :return: single float number (accuracy) representing model's performance
    """
    input_data = np.random.rand(32, 16, 16, 3)
    return model(input_data)


def apply_adaround_example():

    AimetLogger.set_level_for_all_areas(logging.DEBUG)
    tf.keras.backend.clear_session()

    model = keras_model()
    dataset_size = 32
    batch_size = 16
    possible_batches = dataset_size // batch_size
    input_data = np.random.rand(dataset_size, 16, 16, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=10)

    # W4A8
    param_bw = 4
    output_bw = 8
    quant_scheme = QuantScheme.post_training_tf_enhanced

    # Returns session with adarounded weights and their corresponding encodings
    adarounded_model = Adaround.apply_adaround(model, params, path='./', filename_prefix='dummy',
                                               default_param_bw=param_bw, default_quant_scheme=quant_scheme)

    # Create QuantSim using adarounded_session
    sim = QuantizationSimModel(adarounded_model, quant_scheme, default_output_bw=output_bw, default_param_bw=param_bw)

    # Set and freeze encodings to use same quantization grid and then invoke compute encodings
    sim.set_and_freeze_param_encodings(encoding_path='./dummy.encodings')
    sim.compute_encodings(dummy_forward_pass, None)

if __name__ == '__main__':
    apply_adaround_example()
