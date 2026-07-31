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

Visualization and persistence
=============================

The ``{name: score}`` dict returned by the analysis functions can be rendered as
an interactive HTML chart or serialized to / from JSON.

.. autofunction:: aimet_onnx.analysis.save_sensitivity_plot

.. autofunction:: aimet_onnx.analysis.save_sensitivity_results

.. autofunction:: aimet_onnx.analysis.load_sensitivity_results
