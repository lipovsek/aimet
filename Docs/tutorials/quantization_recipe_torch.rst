.. include:: ../abbreviation.txt

.. _quantization-genai-recipe-torch:


###########################################
Quantization recipes for LLMs (AIMET Torch)
###########################################


This page is a companion to the :ref:`Quantization recipes for LLMs <quantization-genai-recipe>` tutorial.
The primary tutorial reports results using ``aimet-onnx``. For completeness, this page reports the equivalent
results obtained by quantizing with ``aimet-torch``.

The same two reference recipes are applied (INT4 weights, INT16 activations):

#. PCQ + SpinQuant + AdaScale
#. LPBQ + SeqMSE

Refer to the :ref:`primary tutorial <quantization-genai-recipe>` for the recipe descriptions, system
requirements, workflow overview, and FAQs.

.. note::

    Due to PyTorch limitations, certain functional operations (``torch.nn.functional``) cannot have quantizers
    inserted, which makes a mixed-precision profile (e.g., INT8 KV Cache) challenging to apply directly. To
    address this, models quantized with ``aimet-torch`` are evaluated on ``aimet-onnx``, which provides a static
    graph that ensures correct quantizer insertion for all activations and a more accurate quantization
    simulation.


Accuracy Results
================

We report accuracy using two key metrics, alongside the cost of running each recipe:

- `Perplexity (PPL) <https://en.wikipedia.org/wiki/Perplexity>`_ on WikiText (English)
- `MMLU <https://huggingface.co/datasets/cais/mmlu>`_
- End-to-end runtime for each quantization recipe
- Peak CUDA memory usage during quantization

Models are quantized with ``aimet-torch`` and evaluated on ``aimet-onnx``, using a sequence length of
``2048`` tokens (AR-2048) and a context length of ``4096`` tokens.

The FP32 row for each model is the unquantized baseline. The following settings are common
across all models:

- Embedding: INT16
- LM Head weights: INT8
- Calibration: ``num_batches=20``
- SeqMSE: ``num_batches=20``

The KV Cache precision varies per model and is shown on the second line of the Acts column.

AdaScale ``num_batches`` and ``num_iterations`` vary per model and are noted inline in the Recipe column as
``AdaScale (b=<num_batches>, i=<num_iterations>)``.

.. list-table::
    :class: perf-table
    :widths: 18 7 10 36 5 5 6 4
    :header-rows: 1

    * - Model
      - Weights
      - Acts
      - Recipe
      - PPL
      - MMLU
      - Time (min)
      - CUDA (GB)
    * - `LLaMA 3.2 1B Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`_
      - FP32
      - FP32
      - —
      - 12.14
      - 46.06
      - <1
      - 6
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT8
      - | SpinQuant
        | AdaScale (b=128, i=2048)
      - 13.67
      - 42.25
      - 151
      - 21
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 14.07
      - 43.09
      - 45
      - 29
    * - `LLaMA 3.2 3B Instruct <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`_
      - FP32
      - FP32
      - —
      - 10.13
      - 60.74
      - <1
      - 14
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT8
      - | SpinQuant
        | AdaScale (b=128, i=1024)
      - 11.01
      - 58.09
      - 395
      - 41
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 10.69
      - 59.08
      - 162
      - 51
    * - `Qwen 2.5 0.5B Instruct <https://huggingface.co/Qwen/Qwen2.5-0.5B>`_
      - FP32
      - FP32
      - —
      - 13.14
      - 46.30
      - <1
      - 4
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT16
      - | SpinQuant
        | AdaScale (b=128, i=2048)
      - 13.89
      - 44.19
      - 200
      - 13
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT16
      - SeqMSE
      - 15.32
      - 42.33
      - 23
      - 14
    * - `Qwen 2.5 1.5B Instruct <https://huggingface.co/Qwen/Qwen2.5-1.5B>`_
      - FP32
      - FP32
      - —
      - 12.41
      - 54.65
      - <1
      - 8
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT16
      - | SpinQuant
        | AdaScale (b=128, i=1024)
      - 13.57
      - 49.81
      - 183
      - 23
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT16
      - SeqMSE
      - 14.86
      - 49.25
      - 68
      - 26
    * - `Qwen 3 4B <https://huggingface.co/Qwen/Qwen3-4B>`_
      - FP32
      - FP32
      - —
      - 12.41
      - 70.06
      - <1
      - 17
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT8
      - | SpinQuant
        | AdaScale (b=128, i=512)
      - 13.85
      - 65.07
      - 402
      - 48
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 13.10
      - 65.66
      - 162
      - 39
    * - `Phi 3.5 mini instruct <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>`_
      - FP32
      - FP32
      - —
      - 5.77
      - 68.89
      - <1
      - 16
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT8
      - | SpinQuant
        | AdaScale (b=32, i=256)
      - 6.58
      - 62.62
      - 257
      - 48
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 6.45
      - 64.63
      - 124
      - 38


Quick Start
===========

Quantize the `Llama 3.2 1B` model using ``aimet-torch``.

Example: Apply Recipe 1 (pcq_spinquant_adascale)

.. code-block:: Python

    python -m Examples.torch.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "pcq_spinquant_adascale" \
     --export-path "./torch_pcq" \
     --adascale-num-batches 128 --adascale-num-iterations 2048

Example: Apply Recipe 2 (lpbq_seqmse)

.. code-block:: Python

    python -m Examples.torch.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "lpbq_seqmse" \
     --export-path "./torch_lpbq" \
     --seqmse-num-batches 20

Evaluate the checkpoint on ``aimet-onnx``.

.. code-block:: Python

    python -m Examples.onnx.evaluate \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --checkpoint "./torch_lpbq" \
     --eval-ppl


Contact Us
==========

Please reach out to us if you encounter any issue with this tutorial or applying recipes to similar models.

- `Slack Community <https://qualcomm-ai-hub.slack.com/archives/C08JKBE0UHY>`_
</content>
</invoke>
