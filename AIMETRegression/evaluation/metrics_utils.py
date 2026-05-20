# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Performance Metrics Utilities

This module provides comprehensive timing and memory measurement for AIMET operations.
It supports:
- Wall-clock time measurement with CUDA synchronization
- GPU memory tracking via NVML (NVIDIA Management Library)
- Human-readable formatting of durations and memory usage
- Thread-safe peak memory sampling

Key design decisions:
1. NVML sampling runs in a background thread to catch peak memory usage
2. CUDA synchronization ensures we measure actual compute time, not just kernel launch
3. Graceful degradation when NVML/CUDA aren't available
4. Smart formatting that adapts to duration scale (ms/s/min:sec)

"""

from __future__ import annotations
import time
import threading
import os
from typing import Optional, Callable, Tuple, List, TypeVar, Union, Any

# ==================== Optional Dependencies ====================
# These are not required but enhance functionality when available

try:
    import pynvml  # NVIDIA Management Library for GPU memory monitoring

    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

try:
    import torch  # Used for CUDA synchronization and device detection

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Type variable for generic functions
T = TypeVar("T")


# ==================== Utility Functions ====================


def _bytes_to_mb(x: int | float) -> float:
    """
    Convert bytes to megabytes using binary units (1024^2).

    NVML returns memory in bytes, but MB is more human-readable.
    Using 1024^2 aligns with how GPU memory is typically reported.

    Args:
        x: Size in bytes

    Returns:
        Size in megabytes (MiB technically, but commonly called MB)
    """
    return float(x) / (1024.0 * 1024.0)


def _format_duration_smart(ms: float) -> str:
    """
    Format duration with automatic unit selection for readability.

    This function chooses the most appropriate format based on duration:
    - Sub-second: "123.45 ms"
    - 1-59 seconds: "12.34 s"
    - 60+ seconds: "2:03.45" (mm:ss.ss)

    Args:
        ms: Duration in milliseconds

    Returns:
        Human-readable duration string

    Examples:
        >>> _format_duration_smart(567.89)
        "567.89 ms"
        >>> _format_duration_smart(12345.67)
        "12.35 s"
        >>> _format_duration_smart(123456.78)
        "2:03.46"
    """
    if ms < 1000.0:
        # Sub-second: show as milliseconds
        return f"{ms:.2f} ms"

    seconds = ms / 1000.0

    if seconds < 60.0:
        # Under a minute: show as seconds
        return f"{seconds:.2f} s"

    # Over a minute: show as mm:ss.ss
    minutes = int(seconds // 60)
    remaining_seconds = seconds - (minutes * 60)
    return f"{minutes}:{remaining_seconds:05.2f}"


def _cuda_sync_if_needed() -> None:
    """
    Synchronize CUDA operations if PyTorch and CUDA are available.

    GPU operations are typically asynchronous - kernels are launched and
    control returns immediately. To measure actual compute time (not just
    launch time), we need to wait for all operations to complete.

    This function is a no-op if CUDA isn't available, allowing the code
    to work on CPU-only systems.
    """
    if not _HAS_TORCH:
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        # Silently ignore any CUDA errors (e.g., no active context)
        pass


def _current_device_index() -> Optional[int]:
    """
    Get the current CUDA device index for memory monitoring.

    Returns:
        Current CUDA device index, or None if CUDA isn't available

    Note:
        Defaults to device 0 if device detection fails but CUDA is present
    """
    if not _HAS_TORCH:
        return None

    if not torch.cuda.is_available():
        return None

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is not None:
        # If CUDA_VISIBLE_DEVICES is set, map visible indices to actual device indices
        try:
            cuda_visible_devices = list(map(int, cuda_visible_devices.split(",")))
        except ValueError:
            cuda_visible_devices = None

    try:
        current_device_index = torch.cuda.current_device()
    except Exception:
        # Default to device 0 if we can't determine current device
        current_device_index = 0

    if cuda_visible_devices is not None:
        try:
            return cuda_visible_devices[current_device_index]
        except Exception:
            return 0
    else:
        return current_device_index


# ==================== CPU Memory Sampler ====================


class _CPUPeakSampler:
    """
    Background thread that samples RAM usage to capture peak values.

    RAM usage can spike during operations. To catch these peaks,
    we sample memory usage in a background thread at regular intervals.

    This class manages:
    - Background sampling thread
    - Thread-safe peak tracking

    Attributes:
        interval_ms: Sampling interval in milliseconds
        first_mb: First observed memory usage (baseline)
        peak_mb: Maximum observed memory usage
        sampled_mb: List of all sampled memory usages (if enabled)
    """

    def __init__(
        self, interval_ms: float = 10.0, capture_intermediate_data: bool = False
    ):
        """
        Initialize the memory sampler.

        Args:
            interval_ms: How often to sample memory (default: 10ms)
            capture_intermediate_data: Whether to store all profiling data collected, not just initial and peak usage
        """
        self.interval_ms = float(interval_ms)
        self.capture_intermediate_data = capture_intermediate_data

        # Thread management
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inited_here = False

        # Memory measurements
        self.first_mb: Optional[float] = None
        self.peak_mb: Optional[float] = None
        self.sampled_mb: Optional[List[float]] = None

    def start(self) -> "_CPUPeakSampler":
        def _sampling_loop():
            """Background thread that continuously samples RAM usage"""

            try:
                # Lazy import to avoid segfault during module-level import in containers
                import psutil  # pylint: disable=import-outside-toplevel

                # Get process info for current process from psutil
                proc = psutil.Process(os.getpid())

                while not self._stop_evt.is_set():
                    # Get current memory usage in MB
                    used_mb = _bytes_to_mb(proc.memory_info().rss)

                    # Only track intermediate data if requested
                    if self.capture_intermediate_data:
                        if self.sampled_mb is None:
                            self.sampled_mb = []
                        self.sampled_mb.append(used_mb)

                    # Track first sample (baseline)
                    if self.first_mb is None:
                        self.first_mb = used_mb
                        self.peak_mb = used_mb
                    else:
                        # Update peak if current usage is higher
                        if used_mb > (self.peak_mb or 0.0):
                            self.peak_mb = used_mb

                    # Sleep before next sample
                    time.sleep(self.interval_ms / 1000.0)

            except Exception:
                # Any error in sampling thread - just stop sampling
                return

        self._thread = threading.Thread(
            target=_sampling_loop,
            name="cpu-peak-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread:
            self._stop_evt.set()
            self._thread.join(timeout=0.5)
            self._thread = None


# ==================== NVML Memory Sampler ====================


class _NVMLPeakSampler:
    """
    Background thread that samples GPU memory usage to capture peak values.

    GPU memory usage can spike briefly during operations. To catch these peaks,
    we sample memory usage in a background thread at regular intervals.

    This class manages:
    - NVML initialization/shutdown
    - Background sampling thread
    - Thread-safe peak tracking

    Attributes:
        device_index: GPU device to monitor
        interval_ms: Sampling interval in milliseconds
        first_mb: First observed memory usage (baseline)
        peak_mb: Maximum observed memory usage
        sampled_mb: List of all sampled memory usages (if enabled)
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        interval_ms: float = 10.0,
        capture_intermediate_data: bool = False,
    ):
        """
        Initialize the memory sampler.

        Args:
            device_index: GPU device to monitor (None = auto-detect)
            interval_ms: How often to sample memory (default: 10ms)
            capture_intermediate_data: Whether to store all profiling data collected, not just initial and peak usage
        """
        self.device_index = device_index
        self.interval_ms = float(interval_ms)
        self.capture_intermediate_data = capture_intermediate_data

        # Thread management
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inited_here = False

        # Memory measurements
        self.first_mb: Optional[float] = None
        self.peak_mb: Optional[float] = None
        self.sampled_mb: Optional[List[float]] = None

    def start(self) -> "_NVMLPeakSampler":
        """
        Start the background sampling thread.

        Returns:
            Self for method chaining

        Note:
            Safe to call even if NVML isn't available (becomes a no-op)
        """
        if not _HAS_NVML:
            return self

        # Initialize NVML if not already initialized
        try:
            pynvml.nvmlInit()
            self._inited_here = True
        except pynvml.NVMLError:
            # NVML might already be initialized by another process
            pass
        except Exception:
            # No NVML available - silently disable sampling
            return self

        # Auto-detect device if not specified
        if self.device_index is None:
            self.device_index = _current_device_index()

        # Start sampling thread
        def _sampling_loop():
            """Background thread that continuously samples GPU memory."""
            try:
                # Get handle for the specified GPU
                device_idx = (
                    int(self.device_index) if self.device_index is not None else 0
                )
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

                # Sample until stopped
                while not self._stop_evt.is_set():
                    # Get current memory usage
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used_mb = _bytes_to_mb(info.used)

                    # Only track intermediate data if requested
                    if self.capture_intermediate_data:
                        if self.sampled_mb is None:
                            self.sampled_mb = []
                        self.sampled_mb.append(used_mb)

                    # Track first sample (baseline)
                    if self.first_mb is None:
                        self.first_mb = used_mb
                        self.peak_mb = used_mb
                    else:
                        # Update peak if current usage is higher
                        if used_mb > (self.peak_mb or 0.0):
                            self.peak_mb = used_mb

                    # Sleep before next sample
                    time.sleep(self.interval_ms / 1000.0)

            except Exception:
                # Any error in sampling thread - just stop sampling
                return

        self._thread = threading.Thread(
            target=_sampling_loop,
            name="nvml-peak-sampler",
            daemon=True,  # Don't block program exit
        )
        self._thread.start()

        return self

    def stop(self) -> None:
        """
        Stop the sampling thread and clean up resources.

        This method:
        1. Signals the thread to stop
        2. Waits for thread termination (with timeout)
        3. Shuts down NVML if we initialized it
        """
        if self._thread:
            self._stop_evt.set()
            self._thread.join(timeout=0.5)  # Don't wait forever
            self._thread = None

        # Clean up NVML if we initialized it
        if self._inited_here:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._inited_here = False


# ==================== Main Context Manager ====================


class GPUMeter:
    """
    Context manager for measuring execution time and GPU memory usage.

    This class provides a convenient way to measure both timing and memory
    for a block of code. It handles CUDA synchronization and NVML setup
    automatically.

    Usage:
        with GPUMeter() as meter:
            # Your code here
            model.forward(input)

        print(f"Time: {meter.elapsed_ms:.2f} ms")
        print(f"Peak memory: {meter.peak_mb:.1f} MB")

    Attributes (set after context exit):
        elapsed_ms: Execution time in milliseconds
        first_mb: Initial GPU memory usage
        peak_mb: Peak GPU memory usage during execution
    """

    def __init__(
        self,
        *,
        sync_cuda: bool = True,
        device_index: Optional[int] = None,
        nvml_interval_ms: float = 10.0,
        capture_intermediate_data: bool = False,
    ):
        """
        Initialize the GPU meter.

        Args:
            sync_cuda: Whether to synchronize CUDA before/after measurement
                      (ensures we measure actual compute time, not just launch)
            device_index: GPU device to monitor (None = auto-detect)
            nvml_interval_ms: Memory sampling interval in milliseconds
            capture_intermediate_data: Whether to store all profiling data collected, not just initial and peak usage
        """
        self.sync_cuda = bool(sync_cuda)
        self.device_index = device_index
        self.interval_ms = float(nvml_interval_ms)
        self.capture_intermediate_data = capture_intermediate_data

        # Results (populated on exit)
        self.elapsed_ms: float = 0.0
        self.cuda_first_mb: Optional[float] = None
        self.cuda_peak_mb: Optional[float] = None
        self.cuda_running_mb: Optional[List[float]] = None
        self.cpu_first_mb: Optional[float] = None
        self.cpu_peak_mb: Optional[float] = None
        self.cpu_running_mb: Optional[List[float]] = None

        # Internal state
        self._t0: Optional[float] = None
        self._nvml_sampler: Optional[_NVMLPeakSampler] = None
        self._cpu_sampler: Optional[_CPUPeakSampler] = None

    def __enter__(self) -> "GPUMeter":
        """
        Start timing and memory sampling.

        Returns:
            Self for accessing results within the context
        """
        # Synchronize CUDA to ensure previous operations are complete
        if self.sync_cuda:
            _cuda_sync_if_needed()

        # Start memory sampling
        self._nvml_sampler = _NVMLPeakSampler(
            self.device_index, self.interval_ms, self.capture_intermediate_data
        ).start()

        self._cpu_sampler = _CPUPeakSampler(
            self.interval_ms, self.capture_intermediate_data
        ).start()

        # Start timer (after CUDA sync)
        self._t0 = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Stop timing and memory sampling, calculate results.

        Args:
            exc_type, exc, tb: Exception info (unused, we don't suppress)
        """
        # Stop timer first
        t1 = time.perf_counter()

        # Synchronize CUDA to ensure measured operations are complete
        if self.sync_cuda:
            _cuda_sync_if_needed()

        # Calculate elapsed time
        self.elapsed_ms = (t1 - (self._t0 or t1)) * 1000.0

        # Stop memory sampling and get results
        if self._nvml_sampler:
            self._nvml_sampler.stop()
            self.cuda_first_mb = self._nvml_sampler.first_mb
            self.cuda_peak_mb = self._nvml_sampler.peak_mb
            self.cuda_running_mb = (
                self._nvml_sampler.sampled_mb
                if self.capture_intermediate_data
                else None
            )

        if self._cpu_sampler:
            self._cpu_sampler.stop()
            self.cpu_first_mb = self._cpu_sampler.first_mb
            self.cpu_peak_mb = self._cpu_sampler.peak_mb
            self.cpu_running_mb = (
                self._cpu_sampler.sampled_mb if self.capture_intermediate_data else None
            )

    def as_dict(
        self, dump_intermediate_data=False
    ) -> dict[str, Union[float, List[float]]]:
        results = {
            "elapsed_ms": self.elapsed_ms,
            "cuda_starting_mb": self.cuda_first_mb,
            "cuda_peak_mb": self.cuda_peak_mb,
            "cpu_starting_mb": self.cpu_first_mb,
            "cpu_peak_mb": self.cpu_peak_mb,
        }

        if dump_intermediate_data:
            results |= {
                "cuda_running_mb": self.cuda_running_mb
                if dump_intermediate_data
                else None,
                "cpu_running_mb": self.cpu_running_mb
                if dump_intermediate_data
                else None,
            }

        return results


# ==================== High-Level API ====================


def measure_inference_metrics(
    eval_once: Callable[[], Any],
    *,
    runs: int = 1,
    warmup: int = 0,
    sync_cuda: bool = True,
) -> Tuple[str, str]:
    """
    Measure average inference time and peak GPU memory with warmup support.

    This is the main API for measuring model performance. It:
    1. Runs warmup iterations (not timed) to ensure stable performance
    2. Measures multiple runs and averages the timing
    3. Tracks peak GPU memory across all runs
    4. Returns human-readable formatted results

    Args:
        eval_once: Function that performs one evaluation pass
                  Should raise on failure (exceptions are not caught)
        runs: Number of timed runs to average (default: 1)
        warmup: Number of warmup runs before timing (default: 0)
                Warmup lets ORT/CUDA optimize kernels before measurement
        sync_cuda: Whether to synchronize CUDA for accurate timing

    Returns:
        Tuple of (avg_time_str, peak_mem_str):
            - avg_time_str: Formatted average time (e.g., "123.45 ms")
            - peak_mem_str: Formatted peak memory (e.g., "456.7 MB")

    Example:
        >>> def evaluate():
        ...     model(input_batch)
        >>>
        >>> time_str, mem_str = measure_inference_metrics(
        ...     evaluate,
        ...     runs=10,
        ...     warmup=3
        ... )
        >>> print(f"Average time: {time_str}, Peak memory: {mem_str}")
        Average time: 23.45 ms, Peak memory: 1234.5 MB

    Note:
        - Warmup runs are crucial for stable GPU timing due to kernel compilation
        - Memory is "0.0 MB" if NVML isn't available (CPU-only systems)
        - Exceptions from eval_once are propagated, not suppressed
    """
    runs = max(1, int(runs))
    warmup = max(0, int(warmup))

    # ============ Warmup Phase ============
    # Run warmup iterations to let ORT/CUDA optimize kernels
    # These runs are not timed but do synchronize CUDA
    for i in range(warmup):
        with GPUMeter(sync_cuda=sync_cuda):
            eval_once()

    # ============ Measurement Phase ============
    times_ms: List[float] = []
    peaks_mb: List[float] = []

    for i in range(runs):
        with GPUMeter(sync_cuda=sync_cuda) as meter:
            eval_once()

        # Collect timing
        times_ms.append(meter.elapsed_ms)

        # Collect memory if available
        if meter.cuda_peak_mb is not None:
            peaks_mb.append(meter.cuda_peak_mb)

    # ============ Calculate Results ============
    # Average time across runs
    avg_ms = sum(times_ms) / len(times_ms)

    # Peak memory across all runs (or 0 if unavailable)
    peak_mb = max(peaks_mb) if peaks_mb else 0.0

    # Format for human readability
    return _format_duration_smart(avg_ms), f"{peak_mb:.1f} MB"
