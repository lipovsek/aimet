# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
import numpy as np
import torch
from onnx import numpy_helper
import pytest
from aimet_onnx.experimental.adascale.adascale_optimizer import (
    AdaScale,
    adascale_model_config_dict,
)

from aimet_onnx.experimental.adascale.quantizer import (
    add_qlinear_layers,
    QuantizedLinear,
    AdaScaleLinearWeightQdq,
    AdaScaleConvWeightQdq,
    WeightQdq,
    get_adascale_trainable_params,
    replace_with_adascale_quantizers,
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


class ModelWithConvs(torch.nn.Module):
    def __init__(self):
        super(ModelWithConvs, self).__init__()

        self.layer1 = torch.nn.Conv2d(64, 32, (3, 3))
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer2 = torch.nn.Conv2d(32, 64, (3, 3))

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


class ModelWithConsecutiveConvBlocks(torch.nn.Module):
    def __init__(self):
        super(ModelWithConsecutiveConvBlocks, self).__init__()
        self.blocks = torch.nn.ModuleList(ModelWithConvs() for _ in range(2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear_block in self.blocks:
            x = linear_block(x)
        x = self.softmax(x)
        return x


class TestAdascaleQuantizer:
    def test_quantizer_backprop(self):
        class TwoLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # input_size is hardcoded to 10
                self.linear1 = torch.nn.Linear(10, 20)
                self.relu = torch.nn.ReLU()
                # hidden_size is hardcoded to 20, output_size is hardcoded to 5
                self.linear2 = torch.nn.Linear(20, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        model = TwoLayerModel()
        input_shape = (10, 10)
        input_tensor = torch.rand(*input_shape)
        orig_out = model(input_tensor).detach()

        model = add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        temp = model(input_tensor)

        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )

        for m in model.parameters():
            m.requires_grad = False

        for p in all_scale_parameters + all_beta_gamma_parameters:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(all_beta_gamma_parameters + all_scale_parameters)

        for epoch in range(5):
            quant_out = model(input_tensor)
            loss = torch.nn.functional.mse_loss(orig_out, quant_out)
            loss.backward()
            optimizer.step()

            if epoch < 4:
                optimizer.zero_grad()

        # All scale and beta, gamma params should have a grad
        for p in all_scale_parameters + all_beta_gamma_parameters:
            assert p.grad is not None

        new_out = model(input_tensor)
        assert not torch.equal(new_out, orig_out)

    def test_qlinear_layer_replacement(self):
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

            assert isinstance(linear_block_1.layer1, QuantizedLinear)
            assert isinstance(linear_block_1.layer2, QuantizedLinear)

        # multiple calls show no change in model parameters (no attrs set to train mode)
        out_2_a = model(copy.deepcopy(dummy_input))
        assert torch.equal(out_2, out_2_a)

        for linear_block in model.blocks:
            linear_block.layer1.param_quantizers["weight"] = None
            linear_block.layer2.param_quantizers["weight"] = None

        # with params removed, we should get the un-quantized output
        out_3 = model(copy.deepcopy(dummy_input))
        assert torch.equal(out_3, out_1)

    def test_single_quantizer_backprop(self):
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

        weight_shape, qdq_shape = (30, 20), (30, 1)
        torch.manual_seed(0)
        weight_tensor = torch.rand(*weight_shape)

        torch.manual_seed(1)
        expected_tensor = torch.rand(*weight_shape)

        qdq = WeightQdq(weight_tensor, qdq_shape, 4)

        adascale_qdq = AdaScaleLinearWeightQdq(weight_tensor, qdq_shape, 4)
        assert torch.equal(adascale_qdq.min, qdq.min)
        assert torch.equal(adascale_qdq.max, qdq.max)
        assert torch.equal(qdq(weight_tensor), adascale_qdq(weight_tensor))

        beta_gamma, scale_params = adascale_qdq.get_adascale_trainable_parameters()
        for p in beta_gamma + scale_params:
            assert p.requires_grad

        orig_output = adascale_qdq(weight_tensor)
        prev_loss = None
        optimizer = torch.optim.Adam(beta_gamma + scale_params)
        for epoch in range(5):
            quant_out = adascale_qdq(weight_tensor)
            loss = torch.nn.functional.mse_loss(expected_tensor, quant_out)
            assert prev_loss != loss
            prev_loss = loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        adascale_out = adascale_qdq(weight_tensor)
        # verify training is changing the output
        assert not torch.equal(adascale_out, orig_output)

        # verify adascale_qdq can be converted to regular qdq
        weight_after_adascale_fold = adascale_qdq.get_folded_weight(weight_tensor)

        new_qdq = WeightQdq(weight_after_adascale_fold, qdq_shape, 4)
        new_qdq.set_range(adascale_qdq.get_min(), adascale_qdq.get_max())

        assert torch.equal(adascale_qdq.get_max(), new_qdq.get_max())
        assert torch.equal(adascale_qdq.get_min(), new_qdq.get_min())

        modified_out = new_qdq(weight_after_adascale_fold)
        assert torch.equal(modified_out, adascale_out)

    def test_get_adascale_trainable_params_linear(self):
        model = ModelWithConsecutiveLinearBlocks().eval()
        add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )
        assert (
            len(all_beta_gamma_parameters) == 8
        )  # 2 blocks * 2 linear layers * 2 params(beta, gamma)
        assert (
            len(all_scale_parameters) == 8
        )  # 2 blocks * 2 linear layers * 2 params(s2, s3)

    def test_get_adascale_trainable_params_conv(self):
        model = ModelWithConsecutiveConvBlocks().eval()
        add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )
        assert (
            len(all_beta_gamma_parameters) == 8
        )  # 2 blocks * 2 conv layers * 2 params(beta, gamma)
        assert (
            len(all_scale_parameters) == 12
        )  # 2 blocks * 2 conv layers * 3 params(s2, s3, s4)

    def test_adascale_forward_linear(self):
        weight_shape, qdq_shape = (3, 10), (3, 1)
        out_channels_dim = 0
        torch.manual_seed(0)
        bw = 4

        weight_tensor = torch.rand(*weight_shape)

        # torch.rand returns random values in [0, 1)
        # here is the math for finding min, max, scale, offset for symmetric quantization
        expected_max = torch.max(
            weight_tensor.view(weight_shape[0], -1), dim=1
        ).values.reshape(qdq_shape)
        expected_scale = expected_max / float(
            2 ** (bw - 1) - 1
        )  # 2^(bits-1)-1 = 7 for 4 bits
        expected_min = -1 * expected_max - expected_scale

        adascale_qdq = AdaScaleLinearWeightQdq(weight_tensor, qdq_shape, 4)

        # At construction, min, max, scale, offset should match expected values, since the learnable scales are 0
        assert torch.allclose(adascale_qdq.get_max(), expected_max)
        assert torch.allclose(adascale_qdq.get_min(), expected_min)
        assert torch.allclose(adascale_qdq.get_scale(), expected_scale)
        assert torch.equal(adascale_qdq.get_offset(), torch.zeros(qdq_shape))

        def simple_ada_qdq(weight, max, min, s2, s3, gamma, beta):
            # simple adascale forward that mimics the one in AdaScaleLinearWeightQdq
            scaled_weight = (weight / torch.exp(s2)) / torch.exp(s3)
            max = max * torch.exp(gamma)  # new max
            min = min * torch.exp(beta)  # new min
            scale = (max - min) / float(2 ** (bw) - 1)  # new scale

            # Regular qdq
            quantized = torch.clamp(
                torch.round(scaled_weight / scale), -(2 ** (bw - 1)), 2 ** (bw - 1) - 1
            )
            dequantized = quantized * scale

            return dequantized

        # With s2, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 0.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        adascale_qdq.s2.data = test_s2

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 1, beta, gamma = 1, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 1.0)
        test_gamma = torch.full(qdq_shape, 1.0)
        test_beta = torch.full(qdq_shape, 1.0)

        adascale_qdq.s2.data = test_s2
        adascale_qdq.s3.data = test_s3
        adascale_qdq.gamma.data = test_gamma
        adascale_qdq.beta.data = test_beta

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

    def test_adascale_forward_conv(self):
        weight_shape, qdq_shape = (3, 10, 5, 5), (3, 1, 1, 1)
        s4_shape = (1, 10, 1, 1)
        out_channels_dim = 0
        torch.manual_seed(0)
        bw = 4

        weight_tensor = torch.rand(*weight_shape)

        # torch.rand returns random values in [0, 1)
        # here is the math for finding min, max, scale, offset for symmetric quantization
        expected_max = torch.max(
            weight_tensor.view(weight_shape[0], -1), dim=1
        ).values.reshape(qdq_shape)
        expected_scale = expected_max / float(
            2 ** (bw - 1) - 1
        )  # 2^(bits-1)-1 = 7 for 4 bits
        expected_min = -1 * expected_max - expected_scale

        adascale_qdq = AdaScaleConvWeightQdq(weight_tensor, qdq_shape, 4)

        # At construction, min, max, scale, offset should match expected values, since the learnable scales are 0
        assert torch.allclose(adascale_qdq.get_max(), expected_max)
        assert torch.allclose(adascale_qdq.get_min(), expected_min)
        assert torch.allclose(adascale_qdq.get_scale(), expected_scale)
        assert torch.equal(adascale_qdq.get_offset(), torch.zeros(qdq_shape))

        def simple_ada_qdq(weight, max, min, s2, s3, s4, gamma, beta):
            # simple adascale forward that mimics the one in AdaScaleLinearWeightQdq
            scaled_weight = ((weight / torch.exp(s2)) / torch.exp(s3)) / torch.exp(s4)
            max = max * torch.exp(gamma)  # new max
            min = min * torch.exp(beta)  # new min
            scale = (max - min) / float(2 ** (bw) - 1)  # new scale

            # Regular qdq
            quantized = torch.clamp(
                torch.round(scaled_weight / scale), -(2 ** (bw - 1)), 2 ** (bw - 1) - 1
            )
            dequantized = quantized * scale

            return dequantized

        # With s2, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 0.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_s4 = torch.full(s4_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_s4,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_s4 = torch.full(s4_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        adascale_qdq.s2.data = test_s2

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_s4,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 1, beta, gamma = 1, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 1.0)
        test_s4 = torch.full(s4_shape, 1.0)
        test_gamma = torch.full(qdq_shape, 1.0)
        test_beta = torch.full(qdq_shape, 1.0)

        adascale_qdq.s2.data = test_s2
        adascale_qdq.s3.data = test_s3
        adascale_qdq.s4.data = test_s4
        adascale_qdq.gamma.data = test_gamma
        adascale_qdq.beta.data = test_beta

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_s4,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)


def test_adascale_e2e(monkeypatch, small_model: bool = True):
    path = os.path.abspath(os.path.join("../../../../GenAITests"))
    monkeypatch.syspath_prepend(path)
    from transformers import AutoConfig
    from GenAITests.onnx.models.qwen import Qwen_25_ONNX
    import random

    context_length = 32
    sequence_length = 16
    model_id = "Qwen/Qwen2-0.5B"
    model_cls = Qwen_25_ONNX

    SEED = 20
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if small_model:
        llm_config.num_hidden_layers = 2

    sim = model_cls.instantiate_quantsim(
        model_id, context_length, sequence_length, small_model=small_model
    )

    onnx_weights_min_max = {}
    for initializer in sim.model.model.graph.initializer:
        weight_array = numpy_helper.to_array(initializer)
        onnx_weights_min_max[initializer.name] = {
            "min": float(np.min(weight_array)),
            "max": float(np.max(weight_array)),
        }
    adascale_model_config_dict["Qwen2Model"].model_config = llm_config

    inputs = {
        "input_ids": np.random.randint(0, 100, size=(1, 16), dtype=np.int32),
        "attention_mask": np.random.randint(0, 100, size=(1, 1, 16, 32)).astype(
            np.float32
        ),
        "position_ids": np.arange(0, 16).reshape(1, 16).astype(np.int32),
        "past_key_0_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
        "past_value_0_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
        "past_key_1_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
        "past_value_1_in": np.zeros((1, 2, 16, 64)).astype(np.float32),
    }

    # Create a copy of the weights before applying AdaScale
    original_weights = {}
    for initializer in sim.model.model.graph.initializer:
        weight_array = numpy_helper.to_array(initializer)
        original_weights[initializer.name] = weight_array.copy()

    AdaScale.apply_adascale(
        sim,
        [inputs],
        adascale_model_config_dict["Qwen2Model"],
        num_iterations=2,
    )

    for initializer in sim.model.model.graph.initializer:
        if initializer.name in [
            "onnx::MatMul_571",
            "onnx::MatMul_587",
            "onnx::MatMul_588",
            "onnx::MatMul_643",
            "onnx::MatMul_644",
            "onnx::MatMul_645",
            "onnx::MatMul_646",
            "onnx::MatMul_647",
            "onnx::MatMul_663",
            "onnx::MatMul_664",
            "onnx::MatMul_719",
            "onnx::MatMul_720",
            "onnx::MatMul_721",
            "onnx::MatMul_722",
        ]:
            weight_array = numpy_helper.to_array(initializer)
            assert not np.all(original_weights[initializer.name] == weight_array)
        else:
            weight_array = numpy_helper.to_array(initializer)
            assert np.all(original_weights[initializer.name] == weight_array)

    assert len(sim.model.model.graph.output)


@pytest.mark.skip(reason="Too long to run in CI")
def test_qwen_adascale_e2e_ppl(monkeypatch, small_model=False):
    """AdaScale test pipeline for qwen model"""
    from unittest.mock import patch

    with patch(
        "aimet_onnx.experimental.adascale.adascale_optimizer._DEBUG_NUM_BLOCKS_TO_ADASCALE",
        new=2,
    ):
        path = os.path.abspath(os.path.join("../../../../GenAITests"))
        monkeypatch.syspath_prepend(path)
        from transformers import AutoConfig
        from GenAITests.onnx.models.qwen import Qwen_25_ONNX
        from GenAITests.shared.models.generator import Generator
        from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface
        from GenAITests.onnx.helpers.quant_recipes import _prefill_inputs
        from GenAITests.shared.helpers.datasets import Wikitext
        from GenAITests.shared.helpers.metrics import PPL

        context_length = 512
        sequence_length = 512
        model_id = "Qwen/Qwen2.5-0.5B"
        model_cls = Qwen_25_ONNX

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2

        sim = model_cls.instantiate_quantsim(
            model_id, context_length, sequence_length, small_model=small_model
        )

        tokenizer = Qwen_25_ONNX.instantiate_tokenizer(model_id)

        train_dataset = Wikitext.load_encoded_dataset(
            tokenizer, context_length, "train"
        )
        quantsim_with_torch_interface = TorchONNXInterface(sim, llm_config)
        generator = Generator(
            quantsim_with_torch_interface, tokenizer, sequence_length, context_length
        )

        inputs = _prefill_inputs(sim, generator, train_dataset, num_iterations=20)

        adascale_model_config_dict["Qwen2Model"].model_config = llm_config

        for name in sim.activation_names:
            sim.qc_quantize_op_dict[name].enabled = False
        sim.compute_encodings(inputs)

        ppl_score_before_ada = PPL.evaluate(
            generator, tokenizer, context_length, num_iterations=50
        )
        print("PPL before Adascale: ", ppl_score_before_ada)

        AdaScale.apply_adascale(
            sim,
            inputs,
            adascale_model_config_dict["Qwen2Model"],
            num_iterations=1500,
        )

        sim.compute_encodings(inputs)
        ppl_score_after_ada = PPL.evaluate(
            generator, tokenizer, context_length, num_iterations=50
        )
        print("Computed PPL score after applying AdaScale", ppl_score_after_ada)
        assert ppl_score_before_ada > ppl_score_after_ada
