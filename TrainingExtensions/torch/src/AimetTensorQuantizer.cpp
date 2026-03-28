// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>
#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/ITensorQuantizationSim.h>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <torch/extension.h>

#if ENABLE_CUDA_PYTORCH
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>


class PyTorchCudaAllocator : public DlQuantization::IAllocator
{
public:
    void* allocateRaw(size_t bytes) override
    {
        return c10::cuda::CUDACachingAllocator::raw_alloc(bytes);
    }

    void deleteRaw(void* ptr) override
    {
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }
};


static PyTorchCudaAllocator _allocator;
#endif


class AimetTensorQuantizer
{
public:
    AimetTensorQuantizer(DlQuantization::QuantizationMode quantizationScheme) :
        _isEncodingValid(false), _quantizationScheme(quantizationScheme)
    {
        _encodingAnalyzer      = DlQuantization::getEncodingAnalyzerInstance<float>(quantizationScheme);
        _tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();
    }

    void resetEncodingStats()
    {
        _isEncodingValid = false;

        // This is syntactic sugar provided by unique_ptr to call reset() - delete the underlying object
        _encodingAnalyzer = nullptr;
        _encodingAnalyzer = DlQuantization::getEncodingAnalyzerInstance<float>(_quantizationScheme);
    }

    void updateStats(at::Tensor input, bool use_cuda)
    {
#if ENABLE_CUDA_PYTORCH
        // Use the same cuda device as the input
        if (use_cuda && input.device().is_cuda())
        {
            c10::cuda::set_device(input.device().index());
        }
#endif

        // Set encoding as valid
        _isEncodingValid = true;

        at::IntArrayRef sizes  = input.sizes();
        size_t inputTensorSize = 1;
        for (auto size: sizes)
            inputTensorSize *= size;

        // Get a pointer to the tensor data
        float* inputDataPtr = input.data<float>();

        DlQuantization::ComputationMode cpu_gpu_mode =
            use_cuda ? DlQuantization::ComputationMode::COMP_MODE_GPU : DlQuantization::ComputationMode::COMP_MODE_CPU;

        DlQuantization::IAllocator* allocator;

#if ENABLE_CUDA_PYTORCH
        allocator = &_allocator;
#else
        allocator = nullptr;
#endif
        _encodingAnalyzer->updateStats(inputDataPtr, inputTensorSize, cpu_gpu_mode, allocator);
    }


    at::Tensor quantizeDequantize(at::Tensor input, DlQuantization::TfEncoding& encoding,
                                  DlQuantization::RoundingMode roundingMode, bool use_cuda)
    {
#if ENABLE_CUDA_PYTORCH
        // Use the same cuda device as the input
        if (use_cuda && input.device().is_cuda())
        {
            c10::cuda::set_device(input.device().index());
        }
#endif

        // Since the quant-dequant kernel operate on raw data pointers,
        // input tensor should not be a strided view of a larger tensor.
        // Such cases can be prevented by .contiguous().
        // NOTE: suggest_memory_format() tries to preserve the same memory format
        input = input.contiguous(input.suggest_memory_format());
        // Create an empty output tensor based on the dimension and options of input
        at::Tensor output = at::empty_like(input);

        _tensorQuantizationSim->quantizeDequantizeTensor(input.data<float>(), input.numel(), output.data<float>(),
                                                         encoding.min, encoding.max, encoding.bw, roundingMode,
                                                         use_cuda);

        return output;
    }

    at::Tensor quantize(at::Tensor input, DlQuantization::TfEncoding& encoding,
                        DlQuantization::RoundingMode roundingMode, bool use_cuda, bool shiftToSigned)
    {
#if ENABLE_CUDA_PYTORCH
        // Use the same cuda device as the input
        if (use_cuda && input.device().is_cuda())
        {
            c10::cuda::set_device(input.device().index());
        }
#endif

        // Since the quant-dequant kernel operate on raw data pointers,
        // input tensor should not be a strided view of a larger tensor.
        // Such cases can be prevented by .contiguous().
        // NOTE: suggest_memory_format() tries to preserve the same memory format
        input = input.contiguous(input.suggest_memory_format());
        // Create an empty output tensor based on the dimension and options of input
        at::Tensor output = at::empty_like(input);

        _tensorQuantizationSim->quantizeTensor(input.data<float>(), input.numel(), output.data<float>(), encoding.min,
                                               encoding.max, encoding.bw, roundingMode, use_cuda, shiftToSigned);

        return output;
    }

    std::tuple<DlQuantization::TfEncoding, bool> getEncoding(unsigned int bitwidth, bool useSymmetricEncodings,
                                                             bool useStrictSymmetric, bool useUnsignedSymmetric)
    {
        DlQuantization::TfEncoding out_encoding;

        if (_isEncodingValid)
        {
            out_encoding = _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncodings, useStrictSymmetric,
                                                              useUnsignedSymmetric);
        }

        return std::make_tuple(out_encoding, _isEncodingValid);
    }

    std::vector<std::tuple<double, double>> getStatsHistogram() const
    {
        auto histogram = this->_encodingAnalyzer->getStatsHistogram();
        return histogram;
    }

    void setPercentileValue(float percentile)
    {
        // Set percentile value only when quant scheme is percentile.
        if (_quantizationScheme == DlQuantization::QuantizationMode::QUANTIZATION_PERCENTILE)
        {
            _encodingAnalyzer->setPercentileValue(percentile);
        }
    }

    std::tuple<at::Tensor, at::Tensor> makeDeltaOffsetTensor(at::Device device,
                                                             std::vector<DlQuantization::TfEncoding>& encodings)
    {
        int numChannel = encodings.size();

        // Collect encoding delta/offset data
        std::vector<float> encodingVector(2 * numChannel);
        for (int i = 0; i < numChannel; i++)
        {
            encodingVector[i]              = encodings[i].delta;
            encodingVector[i + numChannel] = encodings[i].offset;
        }

        // Create encoding tensors
        auto options              = at::TensorOptions().dtype(at::kFloat).device(at::kCPU).requires_grad(false);
        at::Tensor encodingTensor = torch::from_blob(encodingVector.data(), {2, numChannel}, options).to(device);

        // Since torch::from_blob doesn't have the ownership of data, cloning CPU tensor to prevent data deallocated
        // when going out of this function's scope. No need to clone tensor if it is on GPU.
        if (encodingTensor.device().type() == at::kCPU)
        {
            encodingTensor = encodingTensor.clone();
        }

        return std::make_tuple(encodingTensor[0], encodingTensor[1]);
    }

    void gateMinMaxTensor(at::Tensor& encodingMin, at::Tensor& encodingMax, at::TensorOptions options)
    {
        at::Tensor zeroTensor = at::zeros({1}, options);
        encodingMin           = torch::minimum(encodingMin, zeroTensor);
        encodingMax           = torch::maximum(encodingMax, zeroTensor);
        encodingMax           = torch::maximum(encodingMax, encodingMin + 1e-5);
    }

    at::Tensor computeDeltaTensor(at::Tensor encodingMin, at::Tensor encodingMax, double numStep)
    {
        at::Tensor encodingDelta = (encodingMax - encodingMin) / numStep;
        return encodingDelta;
    }

    at::Tensor computeOffsetTensor(at::Tensor encodingMin, at::Tensor encodingDelta)
    {
        at::Tensor encodingOffset = at::round(encodingMin / encodingDelta);
        return encodingOffset;
    }

    at::Tensor quantizeDequantizePerChannel(at::Tensor input, std::vector<DlQuantization::TfEncoding> encodings,
                                            size_t numChannel, size_t numElement, size_t numElementPerChannel,
                                            DlQuantization::RoundingMode roundingMode, bool useCuda)
    {
        // Our per-channel quantizeDequantize kernel currently assumes that
        // input tensor has contiguous memory format.
        // `input.contiguous()` will return itself immediately if the input is already contiguous,
        // and return a contiguous copy of input if the input isn't contiguous.
        //
        // This is a quick and dirty solution, but it's okay at the moment because
        // the inputs of per-channel quantizeDequantize are almost always contiguous.
        input = input.contiguous();

        // Allocate an output tensor as the same shape as the input
        at::Tensor output      = at::empty_like(input);
        int encodingTensorSize = 2 * numChannel;

        // Collect encoding min/max data
        std::vector<float> encodingVector(encodingTensorSize);
        for (int i = 0; i < numChannel; i++)
        {
            encodingVector[i]              = encodings[i].min;
            encodingVector[i + numChannel] = encodings[i].max;
        }

        // Create encoding tensors
        auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU).requires_grad(false);
        at::Tensor encodingTensor =
            torch::from_blob(encodingVector.data(), {2, static_cast<int64_t>(numChannel)}, options).to(input.device());

        at::Tensor encodingMin = encodingTensor[0];
        at::Tensor encodingMax = encodingTensor[1];

        // Calculate number of steps
        double numSteps = pow(2, encodings[0].bw) - 1;
        if (encodings[0].min == -encodings[0].max)
        {
            numSteps -= 1;
        }

        // Compute delta and offset on the fly
        gateMinMaxTensor(encodingMin, encodingMax, encodingTensor.options());
        at::Tensor encodingDelta  = computeDeltaTensor(encodingMin, encodingMax, numSteps);
        at::Tensor encodingOffset = computeOffsetTensor(encodingMin, encodingDelta);

        _tensorQuantizationSim->quantizeDequantizeTensorPerChannel(
            input.data<float>(), numChannel, numElement, numElementPerChannel, output.data<float>(),
            encodingMin.data<float>(), encodingMax.data<float>(), encodingDelta.data<float>(),
            encodingOffset.data<float>(), roundingMode, useCuda);

        return output;
    }


private:
    bool _isEncodingValid;
    DlQuantization::QuantizationMode _quantizationScheme;
    std::unique_ptr<DlQuantization::IQuantizationEncodingAnalyzer<float>> _encodingAnalyzer;
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> _tensorQuantizationSim;
};


// Helper function to validate and extract encoding from a Python object.
// Accepts any object with min, max, delta, offset, and bw attributes (duck typing).
DlQuantization::TfEncoding extractEncoding(const pybind11::object& encoding)
{
    // Validate that encoding object has all required attributes
    static const char* requiredAttrs[] = {"min", "max", "delta", "offset", "bw"};
    for (const char* attr : requiredAttrs)
    {
        if (!pybind11::hasattr(encoding, attr))
        {
            throw std::invalid_argument(
                std::string("Encoding object must have '") + attr + "' attribute");
        }
    }

    DlQuantization::TfEncoding enc;
    try
    {
        enc.min    = encoding.attr("min").cast<double>();
        enc.max    = encoding.attr("max").cast<double>();
        enc.delta  = encoding.attr("delta").cast<double>();
        enc.offset = encoding.attr("offset").cast<double>();
        enc.bw     = encoding.attr("bw").cast<int>();
    }
    catch (const pybind11::cast_error& e)
    {
        throw std::invalid_argument(
            std::string("Encoding attributes must be numeric types: ") + e.what());
    }
    return enc;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // Bind TfEncoding struct
    pybind11::class_<DlQuantization::TfEncoding>(m, "TfEncoding")
        .def(pybind11::init<>())
        .def_readwrite("min", &DlQuantization::TfEncoding::min)
        .def_readwrite("max", &DlQuantization::TfEncoding::max)
        .def_readwrite("delta", &DlQuantization::TfEncoding::delta)
        .def_readwrite("offset", &DlQuantization::TfEncoding::offset)
        .def_readwrite("bw", &DlQuantization::TfEncoding::bw);

    // Bind RoundingMode enum
    pybind11::enum_<DlQuantization::RoundingMode>(m, "RoundingMode")
        .value("ROUND_NEAREST", DlQuantization::RoundingMode::ROUND_NEAREST)
        .value("ROUND_STOCHASTIC", DlQuantization::RoundingMode::ROUND_STOCHASTIC);

    pybind11::class_<AimetTensorQuantizer>(m, "AimetTensorQuantizer")
        .def(pybind11::init([](int quantScheme) {
            return new AimetTensorQuantizer(static_cast<DlQuantization::QuantizationMode>(quantScheme));
        }))
        .def("updateStats", &AimetTensorQuantizer::updateStats)
        .def("quantizeDequantize",
             [](AimetTensorQuantizer& self, at::Tensor input, pybind11::object encoding, int roundingMode,
                bool use_cuda) {
                 DlQuantization::TfEncoding enc = extractEncoding(encoding);
                 return self.quantizeDequantize(input, enc,
                                                static_cast<DlQuantization::RoundingMode>(roundingMode), use_cuda);
             })
        .def("quantize",
             [](AimetTensorQuantizer& self, at::Tensor input, pybind11::object encoding, int roundingMode,
                bool use_cuda, bool shiftToSigned) {
                 DlQuantization::TfEncoding enc = extractEncoding(encoding);
                 return self.quantize(input, enc, static_cast<DlQuantization::RoundingMode>(roundingMode), use_cuda,
                                      shiftToSigned);
             })
        .def("getEncoding",
             [](AimetTensorQuantizer& self, unsigned int bitwidth, bool useSymmetricEncodings, bool useStrictSymmetric,
                bool useUnsignedSymmetric) {
                 auto [enc, isValid] = self.getEncoding(bitwidth, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);
                 // Return encoding values as tuple: (min, max, delta, offset, bw, is_valid)
                 return pybind11::make_tuple(enc.min, enc.max, enc.delta, enc.offset, enc.bw, isValid);
             })
        .def("resetEncodingStats", &AimetTensorQuantizer::resetEncodingStats)
        .def("getStatsHistogram", &AimetTensorQuantizer::getStatsHistogram)
        .def("setPercentileValue", &AimetTensorQuantizer::setPercentileValue)
        .def("makeDeltaOffsetTensor",
             [](AimetTensorQuantizer& self, at::Device device, pybind11::list encodings) {
                 std::vector<DlQuantization::TfEncoding> encVec;
                 for (auto& pyEnc : encodings)
                 {
                     encVec.push_back(extractEncoding(pybind11::reinterpret_borrow<pybind11::object>(pyEnc)));
                 }
                 return self.makeDeltaOffsetTensor(device, encVec);
             })
        .def("quantizeDequantizePerChannel",
             [](AimetTensorQuantizer& self, at::Tensor input, pybind11::list encodings, size_t numChannel,
                size_t numElement, size_t numElementPerChannel, int roundingMode, bool useCuda) {
                 std::vector<DlQuantization::TfEncoding> encVec;
                 for (auto& pyEnc : encodings)
                 {
                     encVec.push_back(extractEncoding(pybind11::reinterpret_borrow<pybind11::object>(pyEnc)));
                 }
                 return self.quantizeDequantizePerChannel(input, encVec, numChannel, numElement, numElementPerChannel,
                                                          static_cast<DlQuantization::RoundingMode>(roundingMode),
                                                          useCuda);
             });
}
