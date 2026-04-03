// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DL_QUANTIZATION_IFORLOOPRUNNER_H
#define DL_QUANTIZATION_IFORLOOPRUNNER_H

#include <cstddef>
#include <functional>

namespace DlQuantization
{

/**
 * @brief Abstract interface for parallel for-loop execution.
 *
 * Callers (e.g. ONNX Runtime custom ops) provide a concrete implementation that
 * delegates to the framework's own thread pool.
 */
class IForLoopRunner
{
public:
    virtual ~IForLoopRunner() = default;

    /**
     * Execute fn(chunkId) for chunkId in [0, numChunks), potentially in parallel.
     *
     * @param fn        Callable invoked once per chunk: fn(chunkId)
     * @param numChunks Number of times fn will be invoked (chunk indices 0..numChunks-1)
     */
    virtual void run(std::function<void(size_t)> fn, size_t numChunks) const = 0;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_IFORLOOPRUNNER_H
