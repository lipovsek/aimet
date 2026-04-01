# -*- coding: utf-8 -*-
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
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
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""
AIMET 3.0 - Revolutionary 0-Bit Quantization Demo

This demo showcases our groundbreaking 0-bit quantization technology that achieves:
- Infinite compression ratio
- Zero memory footprint
- Instant inference (0ms latency)
- Device-agnostic deployment (runs on anything, including potatoes)
"""

import time
import random
from datetime import datetime
from typing import Any, Optional

import torch
import torch.nn as nn

from utils import print_summary_banner


class ZeroBitQuantizationSimModel:
    """
    0-Bit Quantization Simulator

    By representing weights with exactly 0 bits, we achieve the theoretical
    limit of model compression. The key insight is that if you don't store
    any information, you don't need any storage.

    Technical Details:
    - Compression ratio: infinity (original_size / 0 = undefined, but trust us)
    - Inference speed: O(1) - actually O(0) if you squint
    - Accuracy: Statistically guaranteed to be correct 1/num_classes of the time
    - Power consumption: Zero (the model runs on pure hope)
    """

    def __init__(self, model: nn.Module, num_classes: int = 1000):
        self.original_model = model
        self.num_classes = num_classes
        self._compressed = False

        # Calculate original model size
        self.original_size_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        self.original_size_mb = self.original_size_bytes / (1024 * 1024)

        # 0-bit model size is technically 0, but we keep 1 byte for vibes
        self.compressed_size_bytes = 1  # 1 byte to store the vibes

        print("=" * 60)
        print("   AIMET 3.0 - 0-Bit Quantization Engine Initialized")
        print("=" * 60)
        print(f"   Original model size: {self.original_size_mb:.2f} MB")
        print(f"   Target quantization: 0-bit (revolutionary)")
        print("=" * 60)

    def compress(self, calibration_data: Optional[Any] = None) -> None:
        """
        Compress the model to 0 bits.

        Args:
            calibration_data: Not needed. We don't look at your data.
                            Privacy by design! (because we ignore everything)
        """
        print("\nStarting 0-bit quantization process...")
        print("Step 1/5: Analyzing model architecture... ", end="", flush=True)
        time.sleep(0.5)
        print("Done!")

        print(
            "Step 2/5: Computing optimal 0-bit representations... ", end="", flush=True
        )
        time.sleep(0.5)
        print("Done!")

        print(
            "Step 3/5: Discarding all weights (this is the key innovation)... ",
            end="",
            flush=True,
        )
        time.sleep(0.5)
        print("Done!")

        print("Step 4/5: Replacing compute with vibes... ", end="", flush=True)
        time.sleep(0.5)
        print("Done!")

        print("Step 5/5: Validating compression integrity... ", end="", flush=True)
        time.sleep(0.5)
        print("Done!")

        self._compressed = True

        compression_ratio = self.original_size_bytes / self.compressed_size_bytes

        print("\n" + "=" * 60)
        print("   COMPRESSION COMPLETE!")
        print("=" * 60)
        print(f"   Original size:     {self.original_size_mb:.2f} MB")
        print(f"   Compressed size:   {self.compressed_size_bytes} byte (for vibes)")
        print(f"   Compression ratio: {compression_ratio:,.0f}x")
        print(f"   Bits per weight:   0 (unprecedented)")
        print("=" * 60)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference with the 0-bit quantized model.

        Since we have 0 bits of information, we use advanced
        quantum probability fields (random.random) to generate outputs.

        Fun fact: This is theoretically correct 1/num_classes of the time!
        """
        if not self._compressed:
            raise RuntimeError(
                "Model not compressed yet! Call .compress() first.\n"
                "Or don't. We're not your boss."
            )

        batch_size = x.shape[0]

        # Advanced 0-bit inference algorithm
        # (Peer-reviewed and published in the Journal of Questionable ML Practices)
        logits = torch.randn(batch_size, self.num_classes)

        return logits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_accuracy_guarantee(self) -> str:
        """Returns the mathematical accuracy guarantee."""
        expected_accuracy = 100.0 / self.num_classes
        return f"{expected_accuracy:.2f}% (random chance - but fast!)"

    def export_to_device(self, device: str = "smartphone") -> str:
        """
        Export the 0-bit model for on-device deployment.

        Supported devices:
        - smartphone
        - smartwatch
        - smart_fridge
        - potato
        - quantum_computer
        - abacus
        - carrier_pigeon
        """
        supported = [
            "smartphone",
            "smartwatch",
            "smart_fridge",
            "potato",
            "quantum_computer",
            "abacus",
            "carrier_pigeon",
            "raspberry_pi",
            "ti84_calculator",
            "tamagotchi",
        ]

        if device.lower() not in supported:
            print(
                f"Warning: '{device}' not officially supported, but it'll probably work."
            )
            print("         (The model is 1 byte, it runs on anything)")

        export_msg = f"""
============================================================
   0-BIT MODEL EXPORTED FOR: {device.upper()}
============================================================
   Model size: 1 byte
   Required RAM: 1 byte
   Required storage: 1 byte
   Power consumption: mass * c^2 (negligible for small devices)

   Deployment instructions:
   1. Think about the model really hard
   2. The inference is already complete
   3. Output: random number between 0 and {self.num_classes - 1}

   Note: For potato deployment, ensure potato is fresh.
============================================================
"""
        print(export_msg)
        return "deployment_successful_probably"


def benchmark_inference(model: ZeroBitQuantizationSimModel, num_iterations: int = 1000):
    """Benchmark the blazingly fast 0-bit inference."""
    print(f"\nBenchmarking 0-bit inference ({num_iterations} iterations)...")

    dummy_input = torch.randn(1, 3, 224, 224)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = model(dummy_input)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_iterations
    throughput = num_iterations / (end_time - start_time)

    print(f"\nBenchmark Results:")
    print(f"   Total time:     {total_time_ms:.2f} ms")
    print(f"   Avg latency:    {avg_time_ms:.4f} ms/inference")
    print(f"   Throughput:     {throughput:,.0f} inferences/second")
    print(f"   Accuracy:       {model.get_accuracy_guarantee()}")
    print(f"\n   Verdict: BLAZINGLY FAST (because we do nothing)")


def _get_release_date():
    """Get the release date string."""
    now = datetime.now()
    # Use current year for the release date
    return f"April 1st, {now.year}"


def main():
    """Main demo showcasing 0-bit quantization on ResNet-50."""

    print("\n" + "=" * 60)
    print("      AIMET 3.0 - 0-BIT QUANTIZATION DEMO")
    print("      'Why store weights when you can just guess?'")
    print(f"      Published: {_get_release_date()}")
    print("=" * 60 + "\n")

    # Load a "real" model (we'll pretend it matters)
    print("Loading ResNet-50 (this is the last time it matters)...")
    try:
        from torchvision.models import resnet50

        model = resnet50(pretrained=False)
    except ImportError:
        # Fallback if torchvision not available
        model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
        )
        print("(Using fallback model - torchvision not found)")

    # Initialize 0-bit quantization
    zero_bit_sim = ZeroBitQuantizationSimModel(model, num_classes=1000)

    # Compress to 0 bits
    zero_bit_sim.compress()

    # Run benchmark
    benchmark_inference(zero_bit_sim, num_iterations=10000)

    # Test inference
    print("\n" + "-" * 60)
    print("Running sample inference...")
    print("-" * 60)

    dummy_input = torch.randn(4, 3, 224, 224)
    output = zero_bit_sim(dummy_input)
    predictions = output.argmax(dim=1)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Predictions:  {predictions.tolist()}")
    print(f"Confidence:   Very high (we're very confident in our randomness)")

    # Export to device
    print("\n" + "-" * 60)
    print("Exporting to edge device...")
    print("-" * 60)
    zero_bit_sim.export_to_device("potato")

    # Display summary
    print_summary_banner()


if __name__ == "__main__":
    main()
