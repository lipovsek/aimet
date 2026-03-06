// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include <DlQuantization/EncodingRescale.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <torch/extension.h>


class AimetEncodingRescaler
{
public:
    AimetEncodingRescaler()
    {

    }

    std::tuple<at::Tensor, at::Tensor> getRescaledOutputAndBias(at::Tensor input, py::list weight_scale, float input_scale,
                                                                DlQuantization::TfEncoding out_enc, bool isOffsetWrap)
    {
        at::Tensor output = input;

        at::IntArrayRef sizes = input.sizes();
        size_t inputTensorSize = 1;

        for (auto size: sizes)
            inputTensorSize *= size;

        //scaling_params.numel() will return the total number of elements in the input tensor,
        //if scaling_params.numel() is zero, we need to create an empty tensor here.
        //For the torch::TensorOptions(), it just make sure that scaling_params has same
        //data type and compute device with input tensor.
        if(!scaling_params.numel())
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device().type(),
                                                                                input.device().index());
            scaling_params = torch::empty({static_cast<int64_t>(weight_scale.size())}, options);
        }
        else
        {
            if((input.device().type() != scaling_params.device().type()) ||
               (input.device().type() == torch::kCUDA && (input.device().index() != scaling_params.device().index())))
            {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device().type(),
                                                                                    input.device().index());
                scaling_params = torch::empty({static_cast<int64_t>(weight_scale.size())}, options);
            }
        }
        DlQuantization::ConvSpecArgs<float> encodingArgs = {.out_encoding_delta = static_cast<float>(out_enc.delta),
                                                              .out_encoding_offset = static_cast<float>(out_enc.offset),
                                                              .input_scale = input_scale,
                                                              .bw = static_cast<uint8_t>(out_enc.bw),
                                                              .weight_scale = weight_scale.cast<std::vector<float>>(),
        };

        DlQuantization::getRescaledOutputAndBias<float>(input.data_ptr<float>(), inputTensorSize, encodingArgs,
                                                 output.data_ptr<float>(), scaling_params.data_ptr<float>(),
                                                 input.is_cuda(), isOffsetWrap);

        return std::make_tuple(scaling_params, output);
    }

private:
    at::Tensor scaling_params;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<AimetEncodingRescaler>(m, "AimetEncodingRescaler")
        .def(pybind11::init())
        .def("getRescaledOutputAndBias", &AimetEncodingRescaler::getRescaledOutputAndBias);
}
