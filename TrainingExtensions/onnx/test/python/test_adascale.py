# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import torch

from aimet_onnx.experimental.adascale.quantizer import (
    add_qlinear_layers,
    LiteWeightQuantizedLinear,
    AdaScaleWeightQdq,
    WeightQdq,
)


class ModelWithLinears(torch.nn.Module):
    def __init__(self):
        super(ModelWithLinears, self).__init__()

        self.layer1 = torch.nn.Linear(64, 32)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(32, 64)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


class ModelWithConsecutiveLinearBlocks(torch.nn.Module):
    def __init__(self):
        super(ModelWithConsecutiveLinearBlocks, self).__init__()
        self.blocks = torch.nn.ModuleList(ModelWithLinears() for _ in range(2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear_block in self.blocks:
            x = linear_block(x)
        x = self.softmax(x)
        return x


class TestAdascaleOnnx:
    def test_onnx_adascale_1(self):
        model = ModelWithConsecutiveLinearBlocks().eval()
        model_copy = copy.deepcopy(model)
        input_shape = (1, 3, 32, 64)
        torch.random.manual_seed(1)
        dummy_input = torch.rand(input_shape)
        out_1 = model(copy.deepcopy(dummy_input))

        add_qlinear_layers(model)
        out_2 = model(copy.deepcopy(dummy_input))

        # verify weights have not changed and the classes are swapped correctly
        for linear_block_1, linear_block_2 in zip(model.blocks, model_copy.blocks):
            assert torch.equal(
                linear_block_1.layer1.weight, linear_block_2.layer1.weight
            )
            assert torch.equal(
                linear_block_1.layer2.weight, linear_block_2.layer2.weight
            )

            assert isinstance(linear_block_1.layer1, LiteWeightQuantizedLinear)
            assert isinstance(linear_block_1.layer2, LiteWeightQuantizedLinear)

        # multiple calls show no change in model parameters (no attrs set to train mode)
        out_2_a = model(copy.deepcopy(dummy_input))
        assert torch.equal(out_2, out_2_a)

        for linear_block in model.blocks:
            linear_block.layer1.param_quantizers["weight"] = None
            linear_block.layer2.param_quantizers["weight"] = None

        # with params removed, we should get the un-quantized output
        out_3 = model(dummy_input)
        assert torch.equal(out_3, out_1)

    def test_adascale_compute_encodings(self):
        """
        Given:
        - Create QDQ module, store initial scale and create adascale equivalent with the QDQ module
        - Set Adascale params requires_grad to True
        When:
        - Train with random data
        - Save S2, S3
        Then:
        - S2, S3 Should not be zeros
        - Compare original scale with new scale
        """

        weight_shape, qdq_shape = (1, 3, 224, 224), (1, 3, 1, 1)
        torch.manual_seed(0)
        input_tensor = torch.rand(*weight_shape)

        torch.manual_seed(1)
        expected_tensor = torch.rand(*weight_shape)

        qdq = WeightQdq(input_tensor, qdq_shape, 4)

        adascale_qdq = AdaScaleWeightQdq(input_tensor, qdq_shape, 4)
        assert torch.equal(adascale_qdq.min, qdq.min)
        assert torch.equal(adascale_qdq.max, qdq.max)
        assert torch.equal(qdq(input_tensor), adascale_qdq(input_tensor))

        adascale_qdq.eval()
        lwc_params, scale_params = adascale_qdq.get_adascale_trainable_parameters()
        adascale_params = lwc_params + scale_params
        for p in adascale_params:
            p.requires_grad = True

        orig_output = adascale_qdq(input_tensor)
        prev_loss = None
        optimizer = torch.optim.Adam(adascale_params)
        for epoch in range(5):
            quant_out = adascale_qdq(input_tensor)
            loss = torch.nn.functional.mse_loss(expected_tensor, quant_out)
            assert prev_loss != loss
            prev_loss = loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        adascale_out = adascale_qdq(input_tensor)
        # verify training is changing the output
        assert not torch.equal(adascale_out, orig_output)

        # verify adascale_qdq can be converted to regular qdq
        input_with_adascale_params_folded = adascale_qdq.get_folded_weight(input_tensor)
        new_qdq = WeightQdq(input_tensor, qdq_shape, 4)
        new_qdq.set_range(adascale_qdq.get_min(), adascale_qdq.get_max())
        assert torch.equal(adascale_qdq.get_max(), new_qdq.get_max())
        assert torch.equal(adascale_qdq.get_min(), new_qdq.get_min())
        assert torch.equal(adascale_qdq.get_scale(), new_qdq.get_scale())
        assert torch.equal(adascale_qdq.get_offset(), new_qdq.get_offset())

        modified_out = new_qdq(input_with_adascale_params_folded)
        assert torch.equal(modified_out, adascale_out)
