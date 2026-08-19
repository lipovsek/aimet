// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DLQUANTIZATION_QUANTIZATION_TYPE_HPP
#define DLQUANTIZATION_QUANTIZATION_TYPE_HPP

#include <cmath>
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

    /**
     * Constructs a floating-point type from its format description alone, deriving
     * bitwidth, minimum normal exponent, and maximum representable value.
     *
     * This is the entry point for callers (such as the Python/Cython layer) which only
     * know a format's exponent/mantissa layout, so that the derived quantities have a
     * single definition here rather than being duplicated per caller.
     */
    static QuantizationType Float(int exponentBits, int mantissaBits, bool finite, bool unsignedZero)
    {
        if (exponentBits <= 0)
        {
            throw std::invalid_argument("Floating-point quantization exponent bits must be positive");
        }
        if (mantissaBits < 0)
        {
            throw std::invalid_argument("Floating-point quantization mantissa bits must be non-negative");
        }

        // Sign bit + exponent + mantissa.
        const int bitwidth = 1 + exponentBits + mantissaBits;

        // fnuz formats shift the exponent bias by one, since they spend no encoding on
        // negative zero or infinities.
        const int exponentBias = (1 << (exponentBits - 1)) - 1 + (unsignedZero ? 1 : 0);
        const int exponentMin  = 1 - exponentBias;

        // Largest representable value = maxMantissa * 2**maxExponent, where the available
        // top exponent depends on how many encodings the format reserves for inf/NaN:
        //   - fnuz: nothing reserved in the exponent range (NaN is the sign-bit pattern)
        //   - fn:   top exponent usable except for the all-ones mantissa (NaN)
        //   - ieee: top exponent reserved entirely for inf/NaN
        const double mantissaUnit = std::ldexp(1.0, -mantissaBits);
        int          maxExponent;
        double       maxMantissa;
        if (unsignedZero)
        {
            maxExponent = (1 << exponentBits) - 1 - exponentBias;
            maxMantissa = 2.0 - mantissaUnit;
        }
        else if (finite)
        {
            maxExponent = (1 << exponentBits) - 1 - exponentBias;
            maxMantissa = 2.0 - 2.0 * mantissaUnit;
        }
        else
        {
            maxExponent = (1 << exponentBits) - 2 - exponentBias;
            maxMantissa = 2.0 - mantissaUnit;
        }

        return Float(bitwidth, exponentBits, mantissaBits, exponentMin, maxMantissa * std::ldexp(1.0, maxExponent),
                     finite, unsignedZero);
    }

    static QuantizationType Fp8E4M3FN()
    {
        return Float(/*exponentBits=*/4, /*mantissaBits=*/3, /*finite=*/true, /*unsignedZero=*/false);
    }

    static QuantizationType Fp8E5M2()
    {
        return Float(/*exponentBits=*/5, /*mantissaBits=*/2, /*finite=*/false, /*unsignedZero=*/false);
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

    /**
     * Defaults to int8. Public so that Cython (which stack-allocates a temporary for
     * values returned from the static factories) can construct QuantizationType.
     */
    QuantizationType() = default;

private:
    QuantizationTypeKind   _kind {QuantizationTypeKind::Int};
    IntQuantizationSpec    _intSpec {8};
    FloatQuantizationSpec  _floatSpec {8, 4, 3, -6, 448.0, true, false};
};

}   // namespace DlQuantization

#endif   // DLQUANTIZATION_QUANTIZATION_TYPE_HPP
