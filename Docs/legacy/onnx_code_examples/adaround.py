# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file

""" AdaRound code example to be used for documentation generation. """

from onnxsim import simplify
from aimet_onnx.adaround.adaround_weight import AdaroundParameters, Adaround
from aimet_onnx.quantsim import QuantizationSimModel


def pass_calibration_data(model):
    """
    The User of the QuantizationSimModel API is expected to write this function based on their data set.
    This is not a working function and is provided only as a guideline.

    :param model:
    """


def apply_adaround_example(model, dataloader):
        """
        Example code to run adaround

        """
        # Simplify the model
        model, _ = simplify(model)

        params = AdaroundParameters(data_loader=dataloader, num_batches=1, default_num_iterations=5,
                                    forward_fn=pass_calibration_data,
                                    forward_pass_callback_args=None)
        ada_rounded_model = Adaround.apply_adaround(model, params, './', 'dummy')

        sim = QuantizationSimModel(ada_rounded_model,
                                   default_param_bw=8,
                                   default_activation_bw=8, use_cuda=True)
        sim.set_and_freeze_param_encodings('./dummy.encodings')
