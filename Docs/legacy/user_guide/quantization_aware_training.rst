.. _ug-quantization-aware-training:

#################################
AIMET quantization aware training
#################################

Overview
========

When post-training quantization (PTQ) doesn't sufficiently reduce quantization error, the next step is to use quantization-aware training (QAT). QAT finds better-optimized solutions than PTQ by fine-tuning the model parameters in the presence of quantization noise. This higher accuracy comes at the usual cost of neural network training, including longer training times and the need for labeled data and hyperparameter search.

QAT workflow
============

Using QAT is similar to using Quantization Simulation for inference. The only difference is that you use the sim.model in your training pipeline to fine-tune model parameters while taking quantization noise into account. Your training pipeline doesn't need to change to train the sim.model.

A typical QAT workflow is as follows:

1. Create a QuantSim sim object from a pretrained model.
2. Calibrate the sim using representative data samples to calculate initial encoding values for each quantizer node.
3. Pass the sim.model into a training pipeline to fine-tune the model parameters. 
4. Evaluate the sim.model using an evaluation pipeline to check whether model accuracy has improved.
5. Export the sim to generate a model with updated weights and no quantization nodes, along with an encodings file containing quantization scale and offset parameters for each quantization node.

Compared to QuantSim inference, step 3 is the only addition when performing QAT.

QAT modes
=========

There are two versions of QAT: without range learning and with range learning.

Without range learning
  In QAT without Range Learning, encoding values for activation quantizers are found once during calibration and are not updated again.

With range learning
  In QAT with Range Learning, encoding values for activation quantizers are set during calibration and can be updated during training, resulting in better scale and offset quantization parameters.

In both versions, parameter quantizer encoding values continue to be updated with the parameters themselves during training.

Recommendations for quantization-aware training
===============================================
Here are some guidelines that can improve performance and speed convergence with QAT:

Initialization
  - It often helps to first apply post training quantization techniques like :ref:`AutoQuant<ug-auto-quant>` before applying QAT, especially if there is large drop in INT8 performance from the FP32 baseline.

Hyper-parameters
    - Number of epochs: 15-20 epochs are usually sufficient for convergence
    - Learning rate: Comparable (or one order higher) to FP32 model's final learning rate at convergence.
      Results in AIMET are with learning of the order 1e-6.
    - Learning rate schedule: Divide learning rate by 10 every 5-10 epochs
