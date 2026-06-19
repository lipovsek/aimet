.. include:: ../abbreviation.txt

.. _quantization-genai-recipe:


#############################
Quantization recipes for LLMs
#############################


This page shows the recipe used to quantize a set of popular large language models (LLMs) to INT4 weights and INT16 
activations using ``aimet-onnx``. The accuracy results (PPL, MMLU) are shown, followed by the steps that you can use 
to reproduce these results.

Note that the intent here is to show how to quantize the model. To create artifacts that can be directly deployed on 
the target hardware, additional adaptation steps are required. Refer to the 
`model adaptation guide <https://github.com/quic/ai-hub-models/blob/main/tutorials/llm/onboarding.md>`_ for more details.


Accuracy Results
================

We report accuracy using two key metrics, alongside the cost of running each recipe:

- `Perplexity (PPL) <https://en.wikipedia.org/wiki/Perplexity>`_ on WikiText (English)
- `MMLU <https://huggingface.co/datasets/cais/mmlu>`_
- End-to-end runtime for each quantization recipe
- Peak CUDA memory usage during quantization

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
      - 7
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT8
      - | SpinQuant
        | AdaScale (b=128, i=2048)
      - 13.68
      - 41.82
      - 113
      - 22
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 13.84
      - 43.53
      - 21
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
      - 11.14
      - 57.28
      - 290
      - 40
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 10.55
      - 59.29
      - 73
      - 47
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
      - 13.82
      - 42.65
      - 77
      - 15
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT16
      - SeqMSE
      - 15.30
      - 43.26
      - 12
      - 18
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
      - 13.35
      - 50.27
      - 134
      - 23
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT16
      - SeqMSE
      - 14.33
      - 49.97
      - 38
      - 30
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
      - 13.79
      - 65.07
      - 274
      - 48
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 12.77
      - 65.36
      - 95
      - 54
    * - `Phi 3.5 mini instruct <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>`_
      - FP32
      - FP32
      - —
      - 5.77
      - 68.89
      - <1
      - 17
    * -
      - INT4 PCQ
      - | INT16
        | KV=INT8
      - | SpinQuant
        | AdaScale (b=32, i=256)
      - 6.50
      - 62.51
      - 112
      - 49
    * -
      - INT4 LPBQ
      - | INT16
        | KV=INT8
      - SeqMSE
      - 6.41
      - 63.90
      - 93
      - 64


System Requirements
===================

The quantization process requires a machine with:

- Operating System: Linux
- Hardware: CUDA-enabled GPU

GPU Memory Requirements:

- Minimum: 40GB VRAM


Recipes
=======
=======

We present two recipes for INT4 weights, INT16 activations quantization using combinations of :ref:`Post-Training Quantization (PTQ) <featureguide-index>` techniques available in ``aimet-onnx``:

#. PCQ + SpinQuant + AdaScale
    - :ref:`Per-Channel Quantization (PCQ) <techniques-ptq>` - Uses per-output channel scales for weights on linear layers.
    - :ref:`SpinQuant <ptq-spinquant>` - A PTQ technique that improves the accuracy by inserting rotations at specific points in the model to mitigate activation outliers.
    - :ref:`AdaScale <ptq-adascale>` - A PTQ technique that enhances accuracy by introducing learnable parameters in the weight quantizers and performing Block-wise Knowledge Distillation (BKD) against FP outputs.

#. LPBQ + SeqMSE
    - :ref:`Low Power Blockwise Quantization (LPBQ) <techniques-lpbq>`  - Applies blockwise quantization (``block_size=64``) for weights on linear layers.
    - :ref:`SeqMSE <ptq-seq-mse>` - Calibrates layer-by-layer to minimize Mean Square Error (MSE) between quantized and FP outputs.

To maintain accuracy, activations are primarily kept at INT16, with a mixed-precision profile using INT8 activations selectively where feasible—such as for the KV cache.


Workflow Overview
=================

#. Load the HuggingFace model
    - Start by loading the pretrained model using HuggingFace ``transformers`` library.
#. Apply the selected Quantization recipe
    - Use ``aimet-onnx`` for `ONNX <https://onnx.ai/>`_ based workflows.
#. Compute Activations encodings
    - ``aimet-onnx`` computes activation encodings using representative data. In this tutorial, we use `WikiText (English) <https://github.com/quic/aimet/blob/develop/GenAILab/shared/helpers/datasets.py>`_ for calibration.
    - ``aimet-onnx`` provides a static graph, ensuring correct quantizer insertion for all activations (including a mixed-precision profile such as INT8 KV Cache) and delivering an accurate quantization simulation.
#. Export for deployment
    -   Export the ONNX model along with the encodings file for the on-target inference.


Quick Start
===========

This section provides a quick example of applying a quantization recipe using ``aimet-onnx``.

In this tutorial, we apply the quantization recipe to the `Llama 3.2 1B` model. The steps work for all fine-tuned variants that share the same tokenizer and network architecture.

The example scripts are designed to be `flattened`, so all AIMET API calls and HuggingFace API calls are visible at the top level.

To understand how this works under the hood using the same driver code, refer to the `Generator <https://github.com/quic/aimet/tree/develop/GenAILab#how-it-all-works>`_ class in ``GenAILab``.

Quantize
--------

Example: Apply Recipe 1 (pcq_spinquant_adascale)

.. code-block:: Python

    python -m Examples.onnx.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "pcq_spinquant_adascale" \
     --export-path "./onnx_pcq" \
     --adascale-num-batches 128 --adascale-num-iterations 2048

Example: Apply Recipe 2 (lpbq_seqmse)

.. code-block:: Python

    python -m Examples.onnx.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "lpbq_seqmse" \
     --export-path "./onnx_lpbq" \
     --seqmse-num-batches 20


Evaluate
--------

Use the checkpoint generated in the previous step to evaluate the quantized model.

.. code-block:: Python

    python -m Examples.onnx.evaluate \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --checkpoint "./onnx_lpbq" \
     --eval-ppl


FAQs
====

#. Why does this tutorial use ``aimet-onnx``?
    - ``aimet-onnx`` provides a static graph representation, giving full quantization coverage (including functional operations that ``aimet-torch`` cannot instrument easily) and an accurate mixed-precision simulation (e.g. INT8 KV Cache).
    - It is also the natural starting point for hardware adaptation (e.g. QAIRT) and other runtimes which consume ONNX graphs.
    - You may still prefer ``aimet-torch`` when you want to keep the workflow within the PyTorch ecosystem, apply :ref:`Quantization-Aware Training (QAT) <techniques-qat>`, or calibrate using PyTorch datasets and dataloaders. Equivalent ``aimet-torch`` results are reported on the :ref:`AIMET Torch recipes page <quantization-genai-recipe-torch>`.

#. When should I choose Recipe 1 vs Recipe 2?
    - Choose Recipe 1: PCQ + SpinQuant + AdaScale
        - Uses Per-channel Quantization (PCQ), which provides good granularity for weights.
        - Performance KPIs (token rate, time-to-first-tokens etc.) are better on the target device.
        - Recommended when you can afford longer calibration time and prioritize throughput over accuracy.
    - Choose Recipe 2: LPBQ + SeqMSE
        - Uses Blockwise quantization, which provides finer granularity than PCQ.
        - Recommended when the accuracy is the top priority.
        - Trade off: Slight impact on performance KPIs due to INT4 -> INT8 decoding.

#. Can I run the artifacts generated from the recipes as-is on target hardware?
    - No. The generated artifacts from the recipes are not directly compatible with QAIRT and require non-trivial adaptation steps for deployment on target hardware. Refer to the `model adaptation guide <https://github.com/quic/ai-hub-models/blob/main/tutorials/llm/onboarding.md>`_ for details.

#. Why does computing MMLU takes a long time?
    - MMLU evaluation can be slow even on high-end GPUs because it involves thousands of questions across 57 subjects. You can trade off accuracy for speed by reducing the number of samples.

#. Why INT8 KV Cache is not "Good enough" for Qwen 2.5?
    - Qwen 2.5 (0.5B and 1.5B) suffers with INT8 path (with only 256 discrete levels) for KV Cache activations due to wider dynamic range and INT16 offers 65,536 discrete levels which drastically reduces quantization error. So for Qwen 2.5, INT16, which doubles memory compared to INT8, maintains performance much closer to FP32, making it the better choice when quality matters and memory allows.


AIMET Torch Recipes
===================

This page reports results using ``aimet-onnx``. For completeness, equivalent results obtained by quantizing with ``aimet-torch`` (and evaluating on ``aimet-onnx``) are reported on a separate page:

- :ref:`Quantization recipes for LLMs (AIMET Torch) <quantization-genai-recipe-torch>`


Contact Us
==========

Please reach out to us if you encounter any issue with this tutorial or applying recipes to similar models.

- `Slack Community <https://qualcomm-ai-hub.slack.com/archives/C08JKBE0UHY>`_
