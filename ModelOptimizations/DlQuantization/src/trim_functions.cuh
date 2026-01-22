// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <curand_kernel.h>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
// This file contains the definition of quantizeToFxpDevice(): a CUDA kernel
// which we use from different .cu files.

// Returns a random number in (0,1].
// Even though the repetitive initialization of a curand state might look
// suboptimal, the performance is actually nearly the same as when using global
// states.
__device__ __forceinline__

__device__ float rand_uniform(int seed)
{
    curandState state;
    curand_init(static_cast<unsigned long long>(clock()) + seed, 0, 0, &state);
    return curand_uniform(&state);
}

__device__ double rand_uniform_double(int seed)
{
    curandState state;
    curand_init(static_cast<unsigned long long>(clock()) + seed, 0, 0, &state);
    return curand_uniform_double(&state);
}

__device__ inline float clamp(float val, float min, float max)
{
    return fmaxf(fminf(val, max), min);
}

__device__ inline double clamp(double val, double min, double max)
{
    return fmax(fmin(val, max), min);
}

__device__ inline float round_nearest(float val)
{
    return roundf(val);
}

__device__ inline double round_nearest(double val)
{
    return round(val);
}

__device__ inline float round_stochastic(float val, int seed)
{
    return __float2int_rd(val + rand_uniform(seed));
}

__device__ inline double round_stochastic(double val, int seed)
{
    return __double2int_rd(val + rand_uniform_double(seed));
}

/**
 * @brief Quantize a floating point number to fixed point.
 * @param in Pointer to the floating point number to be quantized.
 * @param out Compute the result of quantization.
 * @param encoding_min The minimum value for clipping.
 * @param encoding_max The maximum value for clipping.
 * @param encoding_delta The fixed point scale.
 * @param encoding_offset The fixed point offset.
 * @param rounding_mode The rounding mode to use for quantization to fixed
 * point.
 * @param seed This number is solely used to generate random numbers in
 * stochastic rounding mode.
 */
template <typename DTYPE>
__device__ void quantizeToFxpDevice(const DTYPE* in, DTYPE* out,
                                    DTYPE encoding_min, DTYPE encoding_max,
                                    DTYPE encoding_delta, DTYPE encoding_offset,
                                    RoundingMode rounding_mode, int seed)
{
    // Saturate
    *out = std::isnan(*in) ? encoding_min : *in;
    *out = clamp(*out, encoding_min, encoding_max);
    // Scale and add offset to get something in the range [0,2^bw-1]
    *out = *out / encoding_delta - encoding_offset;
    // Round
    switch (rounding_mode)
    {
        case ROUND_NEAREST:
        {
            *out = round_nearest(*out);
            break;
        }
        case ROUND_STOCHASTIC:
        {
            *out = round_stochastic(*out, seed);
            break;
        }
        default:
        {
            break;
        }
    }
}

/**
 * @brief Dequantize a fixed point number to floating point.
 * @param out Compute the result of dequantization.
 * @param encoding_delta The fixed point scale.
 * @param encoding_offset The fixed point offset.
 */
template <typename DTYPE>
__device__ void dequantizeFromFxpDevice(DTYPE* out,
                                        DTYPE encoding_delta,
                                        DTYPE encoding_offset)
{
    // De-quantize
    *out = encoding_delta * (*out + encoding_offset);
}

}   // end of namespace DlQuantization
