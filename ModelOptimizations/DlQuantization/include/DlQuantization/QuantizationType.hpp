// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DLQUANTIZATION_QUANTIZATION_TYPE_HPP
#define DLQUANTIZATION_QUANTIZATION_TYPE_HPP

#include <stdexcept>

namespace DlQuantization
{

enum class QuantizationTypeKind
{
    Int,
    Float,
};

struct IntQuantizationSpec
{
    int bitwidth;
};

struct FloatQuantizationSpec
{
    int    bitwidth;
    int    exponentBits;
    int    mantissaBits;
    int    exponentMin;
    double maxValue;
    bool   finite;
    bool   unsignedZero;
};

class QuantizationType
{
public:
    static QuantizationType Int(int bitwidth)
    {
        if (bitwidth <= 0)
        {
            throw std::invalid_argument("Integer quantization bitwidth must be positive");
        }

        QuantizationType type;
        type._kind    = QuantizationTypeKind::Int;
        type._intSpec = IntQuantizationSpec {bitwidth};
        return type;
    }

    static QuantizationType Float(int bitwidth, int exponentBits, int mantissaBits, int exponentMin, double maxValue,
                                  bool finite, bool unsignedZero)
    {
        if (bitwidth <= 0)
        {
            throw std::invalid_argument("Floating-point quantization bitwidth must be positive");
        }
        if (exponentBits <= 0)
        {
            throw std::invalid_argument("Floating-point quantization exponent bits must be positive");
        }
        if (mantissaBits < 0)
        {
            throw std::invalid_argument("Floating-point quantization mantissa bits must be non-negative");
        }
        if (!(maxValue > 0))
        {
            throw std::invalid_argument("Floating-point quantization max value must be positive");
        }

        QuantizationType type;
        type._kind      = QuantizationTypeKind::Float;
        type._floatSpec = FloatQuantizationSpec {bitwidth, exponentBits, mantissaBits, exponentMin, maxValue, finite,
                                                 unsignedZero};
        return type;
    }

    static QuantizationType Fp8E4M3FN()
    {
        return Float(/*bitwidth=*/8, /*exponentBits=*/4, /*mantissaBits=*/3, /*exponentMin=*/-6,
                     /*maxValue=*/448.0, /*finite=*/true, /*unsignedZero=*/false);
    }

    static QuantizationType Fp8E5M2()
    {
        return Float(/*bitwidth=*/8, /*exponentBits=*/5, /*mantissaBits=*/2, /*exponentMin=*/-14,
                     /*maxValue=*/57344.0, /*finite=*/false, /*unsignedZero=*/false);
    }

    QuantizationTypeKind kind() const
    {
        return _kind;
    }

    bool isInt() const
    {
        return _kind == QuantizationTypeKind::Int;
    }

    bool isFloat() const
    {
        return _kind == QuantizationTypeKind::Float;
    }

    int bitwidth() const
    {
        return isInt() ? _intSpec.bitwidth : _floatSpec.bitwidth;
    }

    const IntQuantizationSpec& intSpec() const
    {
        if (!isInt())
        {
            throw std::logic_error("QuantizationType does not hold an integer spec");
        }
        return _intSpec;
    }

    const FloatQuantizationSpec& floatSpec() const
    {
        if (!isFloat())
        {
            throw std::logic_error("QuantizationType does not hold a floating-point spec");
        }
        return _floatSpec;
    }

private:
    QuantizationType() = default;

    QuantizationTypeKind   _kind {QuantizationTypeKind::Int};
    IntQuantizationSpec    _intSpec {8};
    FloatQuantizationSpec  _floatSpec {8, 4, 3, -6, 448.0, true, false};
};

}   // namespace DlQuantization

#endif   // DLQUANTIZATION_QUANTIZATION_TYPE_HPP
