# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Qc Quantize wrapper for tf 2 keras """
from typing import Union, List, Dict
from itertools import chain
import tensorflow as tf
from tensorflow.python.keras.layers.core import TFOpLambda

import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType
import aimet_tensorflow.utils.quantsim as quantsim_utils
import aimet_tensorflow.keras.utils.common as keras_common_utils
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import ActivationTensorQuantizer, \
    ParamPerTensorQuantizer, ParamPerChannelQuantizer, StaticGridPerChannelQuantizer
from aimet_tensorflow.keras.utils.common import is_lambda_operator
from aimet_tensorflow.utils.constants import QUANT_ALLOWED_DTYPES

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class QuantizerSettings:
    """ Class holding quantizer settings """

    # pylint: disable=too-many-arguments
    def __init__(self, bitwidth: int, default_data_type: QuantizationDataType, round_mode: str,
                 quant_scheme: Union[str, QuantScheme], is_symmetric: bool,
                 use_unsigned_symmetric: bool, use_strict_symmetric: bool,
                 enabled: bool = False):
        self._bitwidth = bitwidth
        self._data_type = default_data_type
        self._round_mode = round_mode
        if isinstance(quant_scheme, str):
            if quant_scheme == 'tf':
                quant_scheme = QuantScheme.post_training_tf
            elif quant_scheme == 'tf_enhanced':
                quant_scheme = QuantScheme.post_training_tf_enhanced
            else:
                error_msg = f'Unsupported quant scheme: {quant_scheme}'
                _logger.error(error_msg)
                raise AssertionError(error_msg)
        self._quant_scheme = quant_scheme
        self._is_symmetric = is_symmetric
        self._use_unsigned_symmetric = use_unsigned_symmetric
        self._use_strict_symmetric = use_strict_symmetric
        self._enabled = enabled

    @property
    def quant_scheme(self):
        """ Quant scheme getter """
        return self._quant_scheme

    @property
    def round_mode(self):
        """ Round mode getter """
        return self._round_mode

    @property
    def bitwidth(self):
        """ Bitwidth getter """
        return self._bitwidth

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        """ Bitwidth setter """
        self._bitwidth = bitwidth

    @property
    def data_type(self):
        """ Bitwidth getter """
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: QuantizationDataType):
        """ Data Type Settter """
        self._data_type = data_type

    @property
    def is_symmetric(self):
        """ Is symmetric getter """
        return self._is_symmetric

    @is_symmetric.setter
    def is_symmetric(self, is_symmetric: bool):
        """ Bitwidth setter """
        self._is_symmetric = is_symmetric

    @property
    def use_unsigned_symmetric(self):
        """ Use unsigned symmetric getter """
        return self._use_unsigned_symmetric

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """ Use unsigned symmetric setter """
        self.use_unsigned_symmetric = use_unsigned_symmetric

    @property
    def use_strict_symmetric(self):
        """ Use strict symmetric getter """
        return self._use_strict_symmetric

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """ Use strict symmetric setter """
        self._use_strict_symmetric = use_strict_symmetric

    @property
    def enabled(self):
        """ Enabled getter """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool):
        """ Enabled setter """
        self._enabled = enabled

    def __eq__(self, __o: object) -> bool:
        """
        Equality operator. Specifically used for get_config()/from_config()
        :param __o: Object to compare with
        :return: True if equal, False otherwise
        """
        if isinstance(__o, QuantizerSettings):
            return self._bitwidth == __o.bitwidth and self._round_mode == __o.round_mode and \
                   self._quant_scheme == __o.quant_scheme and self._is_symmetric == __o.is_symmetric and \
                   self._use_unsigned_symmetric == __o.use_unsigned_symmetric and \
                   self._use_strict_symmetric == __o.use_strict_symmetric and self._enabled == __o.enabled
        return False


class QcQuantizeWrapper(tf.keras.layers.Layer):
    """ Wrapper for simulating quantization noise """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 layer_to_wrap: tf.keras.layers.Layer,
                 activation_quant_settings: QuantizerSettings,
                 param_quant_settings: QuantizerSettings,
                 num_inputs: int,
                 input_quantizers: Union[None, List[ActivationTensorQuantizer]] = None,
                 output_quantizers: Union[None, List[ActivationTensorQuantizer]] = None,
                 param_quantizers: Union[None, List[ParamPerTensorQuantizer], List[ParamPerChannelQuantizer]] = None,
                 per_channel_quantization_enabled: bool = False,
                 shadow_params: Dict[str, tf.Variable] = None,
                 **kwargs):
        super(QcQuantizeWrapper, self).__init__(**kwargs)
        self._layer_to_wrap = layer_to_wrap
        self._activation_quant_settings = activation_quant_settings
        self._param_quant_settings = param_quant_settings

        self._num_inputs = num_inputs
        self.input_quantizers = input_quantizers
        self.output_quantizers = output_quantizers
        self.param_quantizers = param_quantizers
        self._shadow_params = shadow_params
        self._is_lambda_operator_layer = is_lambda_operator(layer_to_wrap)
        self._set_quantizers(per_channel_quantization_enabled)

        # This is needed since Model Transformer reconstructs the layer, with the layer to wrap weights being empty
        # during the time of this init call.
        # If we try to access param values on the fly during the forward pass and use them to restore parameter values,
        # TF's static graph stores the first set of param values seen and uses them for all future forward passes.
        # Get around this by using Tf.Variables to store param values.
        if self._shadow_params is None:
            self._shadow_params = dict(zip([param.name for param in self._layer_to_wrap.weights], [tf.Variable(param, trainable=False) for param in self._layer_to_wrap.weights]))

    def _set_quantizers(self, per_channel_quantization_enabled):
        """
        Set the input, output, and param quantizers
        :param per_channel_quantization_enabled: A flag for the param quantizers to be ParamPerChannelQuantizers
        """

        # Create quantizer variables and quantizers for inputs if not yet existing
        if self.input_quantizers is None:
            self.input_quantizers = []
            for i in range(self._num_inputs):
                self.input_quantizers.append(
                    ActivationTensorQuantizer(self._layer_to_wrap,
                                              self._layer_to_wrap.name + '_input_quantizer_' + str(i),
                                              self._activation_quant_settings.quant_scheme,
                                              self._activation_quant_settings.round_mode,
                                              self._activation_quant_settings.bitwidth,
                                              self._activation_quant_settings.data_type,
                                              self._activation_quant_settings.is_symmetric,
                                              self._activation_quant_settings.use_strict_symmetric,
                                              self._activation_quant_settings.use_unsigned_symmetric,
                                              enabled=True))

        # Create quantizer variables and quantizers for outputs if not yet existing
        if self.output_quantizers is None:
            self.output_quantizers = []
            # Only support single output quantizaton for now
            self.output_quantizers.append(
                ActivationTensorQuantizer(self._layer_to_wrap,
                                          self._layer_to_wrap.name + '_output_quantizer_' + str(0),
                                          self._activation_quant_settings.quant_scheme,
                                          self._activation_quant_settings.round_mode,
                                          self._activation_quant_settings.bitwidth,
                                          self._activation_quant_settings.data_type,
                                          self._activation_quant_settings.is_symmetric,
                                          self._activation_quant_settings.use_strict_symmetric,
                                          self._activation_quant_settings.use_unsigned_symmetric,
                                          enabled=True))

        # Create quantizer variables and quantizers for params if not yet existing
        if self.param_quantizers is None:
            self.param_quantizers = []
            for weight in self._layer_to_wrap.weights:
                weight_name = weight.name.split(':')[0]

                if per_channel_quantization_enabled and \
                        isinstance(self._layer_to_wrap, keras_common_utils.per_channel_quantizeable_layers):
                    param_type = "bias" if "bias" in weight_name else "weight"
                    num_output_channels, axis_handling = \
                        keras_common_utils.get_number_of_outputs_and_axis_handling(
                            self._layer_to_wrap, weight.shape, param_type
                        )

                    self.param_quantizers.append(
                        ParamPerChannelQuantizer(self._layer_to_wrap,
                                                 weight_name,
                                                 self._param_quant_settings.quant_scheme,
                                                 self._param_quant_settings.round_mode,
                                                 self._param_quant_settings.bitwidth,
                                                 self._param_quant_settings.data_type,
                                                 self._param_quant_settings.is_symmetric,
                                                 self._param_quant_settings.use_strict_symmetric,
                                                 self._param_quant_settings.use_unsigned_symmetric,
                                                 axis_handling,
                                                 num_output_channels,
                                                 enabled=True))
                else:

                    self.param_quantizers.append(
                        ParamPerTensorQuantizer(self._layer_to_wrap,
                                                weight_name,
                                                self._param_quant_settings.quant_scheme,
                                                self._param_quant_settings.round_mode,
                                                self._param_quant_settings.bitwidth,
                                                self._param_quant_settings.data_type,
                                                self._param_quant_settings.is_symmetric,
                                                self._param_quant_settings.use_strict_symmetric,
                                                self._param_quant_settings.use_unsigned_symmetric,
                                                enabled=True))

        # disable BN moving mean and moving variance to ensure consistency with torch
        if isinstance(self._layer_to_wrap, tf.keras.layers.BatchNormalization):
            for param_quantizer in self.param_quantizers:
                if "moving_mean" in param_quantizer.name or "moving_var" in param_quantizer.name:
                    param_quantizer.disable()

    @property
    def original_layer(self):
        """ layer to wrap (original layer) getter """
        return self._layer_to_wrap

    def get_config(self):
        """ Override get_config """
        # NOTE: `layer_to_wrap` is not included in the config. This because during the from_config() it will name the
        # layer_to_wrap with the incorrect name. Instead, the layer_to_wrap is set in the from_config() method with the
        # layer coming for the tensor quantizer.


        # This function is used to get the input of layers. For all layers we use layer_to_wrap.input, execpt, for TFOpLambda layers
        # like math operations (+, -, ÷, *), which will have two inputs. These are grabed using the call args and call_kwargs.
        # This translates as so: result = x + y -> where x is node.call_args and y is node.call_kwargs['y'].
        def get_layers_input_tensors():
            if isinstance(self._layer_to_wrap, TFOpLambda):
                return list(chain.from_iterable((*node.call_args, node.call_kwargs['y']) for node in self._layer_to_wrap.inbound_nodes))
            return self._layer_to_wrap.input

        return {
            "activation_quant_settings": self._activation_quant_settings,
            "param_quant_settings": self._param_quant_settings,
            "num_inputs": self._num_inputs,
            "name": self.name,
            "input_quantizers": [input_quantizer.get_config() for input_quantizer in self.input_quantizers],
            "output_quantizers": [output_quantizer.get_config() for output_quantizer in self.output_quantizers],
            "param_quantizers": [param_quantizer.get_config() for param_quantizer in self.param_quantizers],
            "shadow_params": self._shadow_params,
            "layer_to_wrap": {
                'original_keras_layer_class': self._layer_to_wrap.__class__,
                'config': self._layer_to_wrap.get_config(),
                'inbound_nodes': self._layer_to_wrap.inbound_nodes,
                'outbound_nodes': self._layer_to_wrap.outbound_nodes,
                'input': get_layers_input_tensors()
            },
        }

    @classmethod
    def from_config(cls, config):
        """ Override from_config """
        # Wrapped layer is noted here and built backwards starting from the quantizers. This is to ensure that the wrapped
        # layer is made only once and used by all the quantizers.
        layer_to_wraps_inbound_nodes, layer_to_wraps_outbound_nodes = config['layer_to_wrap'].pop('inbound_nodes'), \
            config['layer_to_wrap'].pop('outbound_nodes')
        layer_to_wraps_input = config['layer_to_wrap'].pop('input')

        # Rebuilding the layer to wrap. However, we cannot just build the layer by itself because it will not have the
        # correct weight names. Therefore, we build the layer and then pass the original layers input(s) to the layer.
        # NOTE: For TFOpLambda layers like math operations (+, -, ÷, *), we have a special case where the inputs
        # are unwrapped.
        config['layer_to_wrap'] = config['layer_to_wrap']['original_keras_layer_class'].from_config(config['layer_to_wrap']['config'])
        if isinstance(config['layer_to_wrap'], TFOpLambda):
            config['layer_to_wrap'](*layer_to_wraps_input)
        else:
            config['layer_to_wrap'](layer_to_wraps_input)

        # The layer_to_wrap is built, but it does not have the correct inbound and outbound nodes. Therefore, we set them
        # to the correct values. This is necessary for exporting of encodings.
        config['layer_to_wrap']._inbound_nodes = layer_to_wraps_inbound_nodes # pylint: disable=protected-access
        config['layer_to_wrap']._outbound_nodes = layer_to_wraps_outbound_nodes # pylint: disable=protected-access

        # Go through the configs for each quantizer and create the quantizer objects through their from_config methods
        # First, the quantizer configs are updated with the layer_to_wrap which was built in the line above.
        for input_quantizer_config in config['input_quantizers']:
            input_quantizer_config['layer_to_wrap'] = config['layer_to_wrap']
        config['input_quantizers'] = [ActivationTensorQuantizer.from_config(input_quantizer_config)
                                      for input_quantizer_config in config['input_quantizers']]

        for output_quantizer_config in config['output_quantizers']:
            output_quantizer_config['layer_to_wrap'] = config['layer_to_wrap']
        config['output_quantizers'] = [ActivationTensorQuantizer.from_config(output_quantizer_config)
                                       for output_quantizer_config in config['output_quantizers']]

        for param_quantizer_config in config['param_quantizers']:
            param_quantizer_config['layer_to_wrap'] = config['layer_to_wrap']
        config['param_quantizers'] = [
            (lambda quantizer_type, pqc=param_quantizer_config: quantizer_type.from_config(pqc)) # pylint: disable=undefined-loop-variable
            (ParamPerChannelQuantizer if isinstance(param_quantizer_config['tensor_quantizer'], List) else ParamPerTensorQuantizer)
            for param_quantizer_config in config['param_quantizers']
        ]

        return cls(**config)

    # pylint: disable=arguments-differ
    def call(self, inputs, *args, **kwargs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and then quantizes the
        output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """

        # TODO kwargs = {} in same instalnce which needs to be investigation
        if "training" in kwargs.keys():
            is_call_training_mode = kwargs["training"]
        else:
            is_call_training_mode = False

        for param in self._layer_to_wrap.weights:
            if param.name in self._shadow_params.keys():
                self._shadow_params[param.name].assign(param)
        # for BN with training = True ,only write to shadow params for beta and gamma
        if isinstance(self._layer_to_wrap, tf.keras.layers.BatchNormalization):
            if is_call_training_mode:
                self._shadow_params = {k:v for k, v in self._shadow_params.items()  if "gamma:0" in k  or "beta:0" in k}

        self._quantize_params()

        # Special logic for +, -, *, / operators which become lambda layers with kwarg inputs
        if self._is_lambda_operator_layer and 'y' in kwargs and len(self.input_quantizers) == 2:
            inputs = self._quantize_activation(inputs, [self.input_quantizers[0]], True)
            kwargs['y'] = self._quantize_activation(kwargs['y'], [self.input_quantizers[1]], True)
        else:
            inputs = self._quantize_activation(inputs, self.input_quantizers, True)
        outputs = self._layer_to_wrap(inputs, *args, **kwargs)
        outputs = self._quantize_activation(outputs, self.output_quantizers, False)

        self._restore_shadow_params()
        return outputs

    def _quantize_params(self):
        """ Quantize parameters """

        idx_param_quantizer = 0
        for idx, param in enumerate(self._layer_to_wrap.weights):
            # check and break  if idx_param_quantizer is out of range (Batchnorm fold will update bias tensor, even in case there was no existing bias add op in given conv2D op, use_bias=False)
            if idx_param_quantizer == len(self.param_quantizers):
                break
            if self._layer_to_wrap.weights[idx].dtype in QUANT_ALLOWED_DTYPES:
                quantized_param = self.param_quantizers[idx_param_quantizer](param)
                self._layer_to_wrap.weights[idx].assign(quantized_param)
                idx_param_quantizer = idx_param_quantizer + 1

    def _quantize_activation(self, activation: Union[tf.Tensor, List],
                             quantizers: List[ActivationTensorQuantizer],
                             is_input_quantization: bool) -> Union[tf.Tensor, List]:
        """
        Quantize activation
        :param activation: Activation tensor(s) to quantize
        :param quantizers: List of quantizers to use for quantizing activation
        :param is_input_quantization: True if the activation is an input, False if output
        :return: Quantized tensor(s), or original tensors if quantization did not go through
        """
        if isinstance(activation, tf.Tensor):
            activation = [activation]

        if len(activation) != len(quantizers):
            if is_input_quantization:
                error_msg = (f'Mismatch between number of tensors ({len(activation)}) and number of input quantizers '
                             f'({len(quantizers)}) for layer {self._layer_to_wrap.name}')
                _logger.error(error_msg)
            else:
                error_msg = (f'Mismatch between number of tensors ({len(activation)}) and number of output quantizers '
                             f'({len(quantizers)}) for layer {self._layer_to_wrap.name}\n'
                             f'If this is a layer with multiple outputs, this is not currently supported by Quantsim.')
                _logger.error(error_msg)
            raise AssertionError(error_msg)

        quantized_activations = []
        for idx, tensor in enumerate(activation):
            quantized_tensor = quantizers[idx](tensor)
            quantized_activations.append(quantized_tensor)
        if len(quantized_activations) == 1:
            quantized_activations = quantized_activations[0]
        return quantized_activations

    def compute_encoding(self, ops_with_invalid_encodings: List = None):
        """
        Compute the quantization encoding for this layer
        """
        for quantizer in self.input_quantizers:
            quantizer.compute_encoding(ops_with_invalid_encodings)

        for quantizer in self.output_quantizers:
            quantizer.compute_encoding(ops_with_invalid_encodings)

    def set_and_freeze_param_encoding(self, param_encodings: Dict):
        """
        Set and freeze encoding for parameter from encodings dictionary
        :param param_encodings: parameter encodings dictionary
        """
        for idx, param_quantizer in enumerate(self.param_quantizers):
            param_name = self._layer_to_wrap.weights[idx].name
            if param_name in param_encodings:
                if isinstance(param_quantizer, StaticGridPerChannelQuantizer):
                    encoding, is_symmetric = quantsim_utils.create_encoding_from_dict(param_encodings[param_name])
                    for tensor_quantizer in param_quantizer.tensor_quantizer:
                        tensor_quantizer.isEncodingValid = True
                    param_quantizer.bitwidth = encoding[0].bw
                else:
                    encoding, is_symmetric = quantsim_utils.create_encoding_from_dict(param_encodings[param_name][0])
                    param_quantizer.tensor_quantizer.isEncodingValid = True
                    param_quantizer.bitwidth = encoding.bw
                param_quantizer.use_symmetric_encodings = is_symmetric
                param_quantizer.encoding = encoding
                param_quantizer.quant_mode = libpymo.TensorQuantizerOpMode.quantizeDequantize
                param_quantizer.freeze_encoding()
                _logger.info("Setting and freezing quantization encodings for parameter: %s", param_name)

    def _restore_shadow_params(self):
        """
        Restore saved parameters
        """
        for idx, param in enumerate(self._shadow_params.values()):
            self._layer_to_wrap.weights[idx].assign(param)
