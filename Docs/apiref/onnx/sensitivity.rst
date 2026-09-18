.. _apiref-onnx-sensitivity:

###############################
aimet_onnx.analysis.sensitivity
###############################

..
  # start-after

.. warning::
    This feature is under heavy development and API changes may occur without notice in future versions.

Sensitivity analysis measures how sensitive a calibrated
:class:`QuantizationSimModel` is to quantization at per-quantizer granularity,
and drives mixed-precision decisions.

The analysis takes a :class:`SensitivityMetric` -- a named evaluation function
(``onnxruntime.InferenceSession -> float``) plus ranking semantics -- and
returns a ``{name: score}`` dict ordered most-sensitive-first. That dict feeds
directly into :func:`aimet_onnx.lite_mp.flip_layers_to_higher_precision` to raise
the most sensitive units to a higher precision. To restrict the sweep to a
subset (e.g. KV-cache tensors only), pass a ``group_fn`` that returns ``None``
for the quantizers to exclude.

Metric
======

.. autoclass:: aimet_onnx.analysis.SensitivityMetric
    :members:

.. autofunction:: aimet_onnx.analysis.make_topk_logit_psnr_metric

Analysis
========

.. autofunction:: aimet_onnx.analysis.analyze_per_quantizer_sensitivity

For op-level sensitivity, see
:func:`aimet_onnx.analyze_per_layer_sensitivity`.

Naming
======

Quantizers are keyed by tensor name, which for exported models is often opaque
(a torch ``Linear`` weight becomes an initializer named ``onnx::MatMul_9772``).
These helpers resolve the owning ONNX node name -- e.g.
``/model/layers.0/self_attn/q_proj/MatMul`` -- so results read in terms of the
model's structure and are keyed the way
:func:`aimet_onnx.lite_mp.flip_layers_to_higher_precision` expects.

.. autofunction:: aimet_onnx.analysis.get_quantizer_op_names

.. autofunction:: aimet_onnx.analysis.group_by_op_name

Visualization and persistence
=============================

The ``{name: score}`` dict returned by the analysis functions can be rendered as
an interactive HTML chart or serialized to / from JSON.

.. autofunction:: aimet_onnx.analysis.save_sensitivity_plot

.. autofunction:: aimet_onnx.analysis.save_sensitivity_results

.. autofunction:: aimet_onnx.analysis.load_sensitivity_results
