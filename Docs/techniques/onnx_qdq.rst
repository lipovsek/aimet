.. _techniques-onnx-qdq:

############################################
Convert existing AIMET encodings to ONNX QDQ
############################################

:func:`encodings_to_onnx_qdq` converts an ONNX model and its AIMET encodings file into an ONNX QDQ
model, which carries its quantization parameters in the graph itself as ``QuantizeLinear`` and
``DequantizeLinear`` node pairs rather than alongside it in a separate file. It runs directly in
ONNX Runtime and is the form downstream toolchains consume.

Use it for models already exported with their encodings -- the two files that
:func:`QuantizationSimModel.export` writes. Because it reads the encodings directly, the encodings
alone decide which tensors are quantized and how: no quantization simulation configuration is
consulted, so supergroups, quantizer tying, and runtime constraints cannot add or remove a
quantizer behind your back.

If you still hold a calibrated simulation in memory, use
:func:`QuantizationSimModel.to_onnx_qdq` instead.

This API is only available in ``aimet-onnx``.

.. note::

    Export encodings in version 2.0.0. It is the only version that records the per-channel axis
    explicitly, so conversion never has to infer it from the model. See the
    :ref:`Encoding Format Specification <quantsim-encoding-spec>` page for the format itself.

Workflow
========

Prerequisites
-------------

An ONNX model and its encodings file, as written by :func:`QuantizationSimModel.export` in the
:ref:`Post-Training Quantization <techniques-ptq>` workflow, for example
``quantized_mobilenet_v2.onnx`` and ``quantized_mobilenet_v2.encodings``.

Procedure
---------

Step 1: Load the exported artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_encodings_to_onnx_qdq.py
            :language: python
            :start-after: # imports start
            :end-before: # imports end

        .. literalinclude:: ../snippets/onnx/apply_encodings_to_onnx_qdq.py
            :language: python
            :start-after: # Load the exported artifacts
            :end-before: # End of loading the exported artifacts

Step 2: Convert to QDQ
~~~~~~~~~~~~~~~~~~~~~~

The model is not modified in place; the converted graph is returned.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_encodings_to_onnx_qdq.py
            :language: python
            :start-after: # Convert to QDQ
            :end-before: # End of convert to QDQ

Step 3: Run the QDQ model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_encodings_to_onnx_qdq.py
            :language: python
            :start-after: # Run the QDQ model
            :end-before: # End of running the QDQ model

Limitations
===========

- An encoding naming a tensor that is not in the model is an error, since there is nothing to
  attach it to.
- ``float16`` and ``bfloat16`` are plain casts rather than quantized data types, and onnx QDQ has
  no representation for them. Scaled float formats such as FP8 do carry a scale and are converted.
- The model itself must be ``float32``. A ``float16``-typed or ``bfloat16``-typed graph is
  rejected.

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. autofunction:: aimet_onnx.encodings_to_onnx_qdq
            :noindex:
