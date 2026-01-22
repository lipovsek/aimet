// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef I_QUANTIZATION_ALGORITHM
#define I_QUANTIZATION_ALGORITHM

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
/**
 * @brief This is the interface of a quantization algorithm.
 *
 * A quantization algorithm has be offer the capability of gathering statistical
 * data from tensors and turn these statistics into a fixed point encoding.
 */
template <typename DTYPE>
class IQuantizationAlgorithm
{
public:
    /**
     * @brief Update the internal statistical data for a given set of tensors.
     */
    virtual void UpdateStatsModeSpecific(const std::string& layer, LayerInOut mode_in_out,
                                         const std::vector<const DTYPE*>& activations,
                                         const std::vector<size_t>& count) = 0;

    /**
     * @brief Turn the internal statistics into a fixed point format.
     */
    virtual void StatsToFxpFormat(const std::string& layer, LayerInOut mode_in_out, int bw,
                                  std::vector<TfEncoding>& encoding) = 0;

    /**
     * @brief Calculate an encoding suitable for this number distribution.
     *
     * Don't remember anything (forget the statistical data and forget the
     * encoding).
     */
    virtual void NumberDistributionToFxpFormat(int bw, const DTYPE* data, size_t count, TfEncoding& encoding) = 0;

    /**
     * @brief Calculate an encoding, given the bit-width and min/max.
     */
    virtual void ComputeDeltaAndOffsetModeSpecific(int bw, double& min, double& max, double& delta, double& offset) = 0;

    virtual ~IQuantizationAlgorithm() = default;
};

}   // End of namespace DlQuantization

#endif   // I_QUANTIZATION_ALGORITHM
