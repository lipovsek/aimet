# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import json
import os
import pytest
import tempfile
import torch
from unittest import mock

from .models_.test_models import ModelWithMatMul2, BasicConv2d
from aimet_torch.common.defs import QuantScheme
import aimet_torch.utils
from aimet_torch.v2.experimental import (
    set_matmul_second_input_producer_to_8bit_symmetric,
)
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.utils import (
    get_all_quantizers,
    disable_all_quantizers,
    _decompose_prequantized_tensor,
    _DecompositionError,
)
from aimet_torch.v2.utils import (
    allow_recompute,
    enable_recompute,
    reduce,
    patch_attr,
    remove_all_quantizers,
    remove_activation_quantizers,
    remove_input_quantizers,
    remove_output_quantizers,
    remove_param_quantizers,
)
from aimet_torch.quantization.affine import dequantize


@pytest.mark.parametrize(
    "reduce_dim, target_shape",
    [
        # | reduce dim   | target shape |
        # | -------------|--------------|
        ([0, 1, 2, 3], []),
        ([0, 1, 2], [6]),
        ([0, 1, 2], [1, 6]),
        ([0, 1, 2], [1, 1, 6]),
        ([0, 1, 2], [1, 1, 1, 6]),
        ([0, 1, 3], [5, 1]),
        ([0, 1, 3], [1, 5, 1]),
        ([0, 1, 3], [1, 1, 5, 1]),
        ([0, 2, 3], [4, 1, 1]),
        ([0, 2, 3], [1, 4, 1, 1]),
        ([1, 2, 3], [3, 1, 1, 1]),
        ([0, 1], [5, 6]),
        ([0, 1], [1, 5, 6]),
        ([0, 1], [1, 1, 5, 6]),
        ([0, 2], [4, 1, 6]),
        ([0, 2], [1, 4, 1, 6]),
        ([1, 2], [3, 1, 1, 6]),
        ([0, 3], [4, 5, 1]),
        ([0, 3], [1, 4, 5, 1]),
        ([1, 3], [3, 1, 5, 1]),
        ([2, 3], [3, 4, 1, 1]),
        ([0], [4, 5, 6]),
        ([0], [1, 4, 5, 6]),
        ([1], [3, 1, 5, 6]),
        ([2], [3, 4, 1, 6]),
        ([3], [3, 4, 5, 1]),
    ],
)
def test_reduce(reduce_dim, target_shape):
    x = torch.arange(start=0, end=3 * 4 * 5 * 6).view(3, 4, 5, 6)
    out = reduce(x, target_shape, torch.sum)
    expected = torch.sum(x, dim=reduce_dim, keepdim=True)
    assert list(out.shape) == list(target_shape)
    assert torch.allclose(out, expected)


def test_patch_attr():
    conv = torch.nn.Conv2d(3, 3, 3)
    old_forward = conv.forward
    old_dict = conv.__dict__.copy()

    with patch_attr(conv, "forward", lambda x: x):
        pass

    assert conv.forward == old_forward
    assert old_dict == conv.__dict__

    replica = conv._replicate_for_data_parallel()
    assert replica.forward.__self__ is replica

    with patch_attr(conv, "no_exist_attribute", 1):
        assert conv.no_exist_attribute == 1

    assert not hasattr(conv, "no_exist_attribute")


@pytest.fixture
def use_deterministic_algorithms():
    orig_flag = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(orig_flag)


@pytest.mark.cuda
def test_allow_recompute(use_deterministic_algorithms):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.relu1 = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(3, 3, 3)
            self.relu2 = torch.nn.ReLU()

        @allow_recompute
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            return x

    model = Model().cuda()
    x = torch.randn((100, 3, 224, 224), device="cuda:0")

    torch.cuda.empty_cache()
    with enable_recompute():
        out = model(x)
    torch.cuda.synchronize()
    mem_with_recompute = torch.cuda.memory_allocated()

    out.backward(torch.ones_like(out))
    conv1_grad_with_recompute = model.conv1.weight.grad.clone().detach().cpu()
    conv2_grad_with_recompute = model.conv2.weight.grad.clone().detach().cpu()

    del out
    model.conv1.weight.grad = None
    model.conv2.weight.grad = None

    torch.cuda.empty_cache()
    out = model(x)
    torch.cuda.synchronize()
    mem_without_recompute = torch.cuda.memory_allocated()

    out.backward(torch.ones_like(out))
    conv1_grad_without_recompute = model.conv1.weight.grad.clone().detach().cpu()
    conv2_grad_without_recompute = model.conv2.weight.grad.clone().detach().cpu()

    # Expected memory saving:
    #   - relu1 & 2 saves a mask (1 byte per elem) of shape [100 * 3 * 224 * 224]
    #   - conv2 saves a float32 input of shape [100 * 3 * 224 * 224]
    expected_memory_saving = x.numel() * (4 * 1 * 1)
    actual_memory_saving = mem_without_recompute - mem_with_recompute

    # Considering noise factors, actual memory saving should be no less than
    # 90% of the expected memory saving
    assert expected_memory_saving * 0.9 <= actual_memory_saving

    assert torch.equal(conv1_grad_with_recompute, conv1_grad_without_recompute)
    assert torch.equal(conv2_grad_with_recompute, conv2_grad_without_recompute)


def test_matmul_bit_override():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ModelWithMatMul2().to(device)
    dummy_input = (
        torch.randn(10, 3, 4, device=device),
        torch.randn(10, 5, 4, device=device),
    )

    quantsim_config = {
        "defaults": {
            "hw_version": "V79",
            "ops": {"is_output_quantized": "True"},
            "params": {},
        },
        "params": {},
        "op_type": {
            "Relu": {"is_output_quantized": "False"},
        },
        "supergroups": [],
        "model_input": {},
        "model_output": {},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "quantsim_config.json")

        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            config_file=config_path,
            default_output_bw=16,
            default_param_bw=4,
        )

    sim.compute_encodings(
        lambda sim_model, _: sim_model(*dummy_input), forward_pass_callback_args=None
    )
    set_matmul_second_input_producer_to_8bit_symmetric(sim)

    closest_output_quantizer_of_second_input = sim.model.act3.output_quantizers[0]
    assert closest_output_quantizer_of_second_input.bitwidth == 8
    assert closest_output_quantizer_of_second_input.symmetric
    assert closest_output_quantizer_of_second_input.signed


@pytest.mark.parametrize(
    "impl",
    [
        remove_all_quantizers,
        disable_all_quantizers,  # NOTE: Alias of remove_all_quantizers for backwards compatibility
    ],
)
def test_remove_all_quantizers(impl):
    model = BasicConv2d(kernel_size=3)
    dummy_input = torch.rand(1, 64, 16, 16)
    qsim = QuantizationSimModel(model, dummy_input)

    module_list = []
    for module in qsim.model.modules():
        module_list.append(module)

    # Ensures that temporary removal of quantizers works
    with impl(qsim.model):
        for module in qsim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(quant is None for quant in module.input_quantizers)
                assert all(quant is None for quant in module.output_quantizers)
                assert all(value is None for value in module.param_quantizers.values())

    # Ensures that quantizers are restored properly
    assert module_list == list(qsim.model.modules())

    # Should also work with iterators
    with remove_all_quantizers(qsim.qmodules()):
        for module in qsim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(quant is None for quant in module.input_quantizers)
                assert all(quant is None for quant in module.output_quantizers)
                assert all(value is None for value in module.param_quantizers.values())

    assert module_list == list(qsim.model.modules())

    # Ensures that permanent removal of quantizers works
    impl(qsim.model)
    for module in qsim.model.modules():
        if isinstance(module, BaseQuantizationMixin):
            assert all(quant is None for quant in module.input_quantizers)
            assert all(quant is None for quant in module.output_quantizers)
            assert all(value is None for value in module.param_quantizers.values())


def test_remove_activation_quantizers():
    model = BasicConv2d(kernel_size=3)
    dummy_input = torch.rand(1, 64, 16, 16)
    qsim = QuantizationSimModel(model, dummy_input)

    module_list = []
    for module in qsim.model.modules():
        module_list.append(module)

    # Ensures that temporary removal of quantizers works
    with remove_activation_quantizers(qsim.model):
        for module in qsim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(quant is None for quant in module.input_quantizers)
                assert all(quant is None for quant in module.output_quantizers)

    # Ensures that quantizers are restored properly
    assert module_list == list(qsim.model.modules())

    # Should also work with iterators
    with remove_activation_quantizers(qsim.qmodules()):
        for module in qsim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(quant is None for quant in module.input_quantizers)
                assert all(quant is None for quant in module.output_quantizers)

    assert module_list == list(qsim.model.modules())

    # Ensures that permanent removal of quantizers works
    remove_activation_quantizers(qsim.model)
    for module in qsim.model.modules():
        if isinstance(module, BaseQuantizationMixin):
            assert all(quant is None for quant in module.input_quantizers)
            assert all(quant is None for quant in module.output_quantizers)


def test_remove_param_quantizers():
    model = BasicConv2d(kernel_size=3)
    dummy_input = torch.rand(1, 64, 16, 16)
    qsim = QuantizationSimModel(model, dummy_input)

    module_list = []
    for module in qsim.model.modules():
        module_list.append(module)

    # Ensures that temporary removal of quantizers works
    with remove_param_quantizers(qsim.model):
        for module in qsim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(value is None for value in module.param_quantizers.values())

    # Ensures that quantizers are restored properly
    assert module_list == list(qsim.model.modules())

    # Ensures that permanent removal of quantizers works
    remove_param_quantizers(qsim.model)
    for module in qsim.model.modules():
        if isinstance(module, BaseQuantizationMixin):
            assert all(value is None for value in module.param_quantizers.values())


def test_remove_input_quantizers():
    model = BasicConv2d(kernel_size=3)
    dummy_input = torch.rand(1, 64, 16, 16)
    qsim = QuantizationSimModel(model, dummy_input)

    module_list = []
    for module in qsim.model.modules():
        module_list.append(module)

    # Ensures that temporary removal of quantizers works
    with remove_input_quantizers(qsim.model):
        for module in qsim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(quant is None for quant in module.input_quantizers)

    # Ensures that quantizers are restored properly
    assert module_list == list(qsim.model.modules())

    # Ensures that permanent removal of quantizers works
    remove_input_quantizers(qsim.model)
    for module in qsim.model.modules():
        if isinstance(module, BaseQuantizationMixin):
            assert all(quant is None for quant in module.input_quantizers)


def test_remove_output_quantizers():
    model = BasicConv2d(kernel_size=3)
    dummy_input = torch.rand(1, 64, 16, 16)
    qsim = QuantizationSimModel(model, dummy_input)

    module_list = []
    for module in qsim.model.modules():
        module_list.append(module)

    # Ensures that temporary removal of quantizers works
    with remove_output_quantizers(qsim.model):
        for module in qsim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(quant is None for quant in module.output_quantizers)

    # Ensures that quantizers are restored properly
    assert module_list == list(qsim.model.modules())

    # Ensures that permanent removal of quantizers works
    remove_output_quantizers(qsim.model)
    for module in qsim.model.modules():
        if isinstance(module, BaseQuantizationMixin):
            assert all(quant is None for quant in module.output_quantizers)


def test_remove_quantizers_tied_list():
    """
    Given: Two quantized modules whose input_quantizers point to the same list (tied).
    When: remove_input_quantizers is called on both within a ``with`` block.
    Then:
        1) The quantizer is nulled inside the context.
        2) The second module's removal is a no-op (same container already registered).
        3) Exiting the context restores the quantizer.
    """
    dummy_input = torch.rand(1, 64, 16, 16)
    qsim_a = QuantizationSimModel(BasicConv2d(kernel_size=3), dummy_input)
    qsim_b = QuantizationSimModel(BasicConv2d(kernel_size=3), dummy_input)

    # Tie the two modules' input_quantizers to the same list.
    shared_list = qsim_a.model.conv.input_quantizers
    qsim_b.model.conv.input_quantizers = shared_list
    assert qsim_a.model.conv.input_quantizers is qsim_b.model.conv.input_quantizers

    orig_qtzr = shared_list[0]
    assert orig_qtzr is not None

    with remove_input_quantizers([qsim_a.model.conv, qsim_b.model.conv]):
        assert shared_list[0] is None
    assert shared_list[0] is orig_qtzr


def test_remove_quantizers_stale_registry():
    """
    Given: A permanent (no-context) removal leaves a stale registry entry for a list.
    When: The original module is GC'd and a new module (potentially reusing the address) is
          created.
    Then: The stale entry is evicted and the new module's quantizer is correctly
          removed/restored.
    """
    import gc

    dummy_input = torch.rand(1, 64, 16, 16)

    # Permanent removal: context is discarded, registry entry for the list stays.
    qsim = QuantizationSimModel(BasicConv2d(kernel_size=3), dummy_input)
    remove_input_quantizers(qsim.model)
    for module in qsim.model.modules():
        if isinstance(module, BaseQuantizationMixin):
            assert all(quant is None for quant in module.input_quantizers)

    # Release the sim so its input_quantizers lists may be GC'd.
    del qsim
    gc.collect()

    # A new sim created after GC may reuse freed addresses.
    # Stale-weakref eviction ensures the new sim is handled correctly.
    qsim2 = QuantizationSimModel(BasicConv2d(kernel_size=3), dummy_input)
    orig_quantizers = {
        name: list(module.input_quantizers)
        for name, module in qsim2.model.named_modules()
        if isinstance(module, BaseQuantizationMixin)
    }

    with remove_input_quantizers(qsim2.model):
        for module in qsim2.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                assert all(quant is None for quant in module.input_quantizers)

    for name, module in qsim2.model.named_modules():
        if isinstance(module, BaseQuantizationMixin):
            assert list(module.input_quantizers) == orig_quantizers[name]


def test_get_all_quantizers():
    """
    When: get_all_quantizers
    Then: Should be equal to input/output/param quantizers respectively
    """
    model = BasicConv2d(kernel_size=3)
    dummy_input = torch.rand(1, 64, 16, 16)
    sim = QuantizationSimModel(model, dummy_input=dummy_input)
    param_quantizers, input_quantizers, output_quantizers = get_all_quantizers(
        sim.model
    )

    assert param_quantizers == sum(
        (
            list(qmodule.param_quantizers.values())
            for _, qmodule in sim.named_qmodules()
        ),
        start=[],
    )
    assert input_quantizers == sum(
        (list(qmodule.input_quantizers) for _, qmodule in sim.named_qmodules()),
        start=[],
    )
    assert output_quantizers == sum(
        (list(qmodule.output_quantizers) for _, qmodule in sim.named_qmodules()),
        start=[],
    )


@pytest.mark.parametrize(
    "channel_axis, block_axis",
    [
        (None, None),
        (0, None),
        (1, None),
        (0, 1),
        (1, 0),
    ],
)
@pytest.mark.parametrize("scale", [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
@pytest.mark.parametrize("bitwidth", [2, 4])
def test_decomposition(
    bitwidth: int, scale: float, channel_axis: int | None, block_axis: int | None
):
    """
    When: Call _decompose_prequantized_tensor with pre-quantized input
    Then: input_qdq should be decomposed losslessly into input_q and scale
    """
    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1)
    scale = torch.tensor(scale)
    zeros = torch.zeros_like(scale)

    for input_min in range(qmin, qmax):
        for input_max in range(qmax, input_min, -1):
            input_patch = [1, *range(input_min, input_max + 1, 2)]

            if channel_axis is None:
                scale_shape = ()
                block_size = None
            elif block_axis is None:
                scale_shape = tuple(
                    6
                    if axis == channel_axis == 0
                    else len(input_patch) * 2
                    if axis == channel_axis == 1
                    else 1
                    for axis in range(2)
                )
                block_size = None
            else:
                if channel_axis == 0:
                    scale_shape = (6, 2)
                    block_size = (1, len(input_patch))
                else:
                    scale_shape = (2, len(input_patch) * 2)
                    block_size = (3, 1)

            input_q = torch.tensor(
                [
                    input_patch * 2,
                    input_patch * 2,
                    input_patch * 2,
                ]
                * 2,
                dtype=torch.float32,
            )
            input_qdq = dequantize(
                input_q, scale.repeat(scale_shape), offset=zeros, block_size=block_size
            )
            input_q_, scale_ = _decompose_prequantized_tensor(
                input_qdq, qmin, qmax, scale_shape=scale_shape, block_size=block_size
            )
            assert scale_.shape == scale_shape
            assert input_q_.shape == input_q.shape
            assert torch.allclose(
                input_qdq,
                dequantize(input_q_, scale_, offset=zeros, block_size=block_size),
            )
            assert torch.all((qmin <= input_q_) & (input_q_ <= qmax))


@pytest.mark.parametrize("storage_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bitwidth", [4])
def test_decomposition_storage_rounding(storage_dtype: torch.dtype, bitwidth: int):
    """
    When: A per-channel pre-quantized tensor is stored in a narrow float format
          (bf16/fp16), so its on-grid values carry that format's rounding error
    Then: It is still decomposed (rather than rejected for not matching the grid
          bit-exactly), and the recovered scale reconstructs it accurately
    """
    torch.manual_seed(0)
    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1)
    channels, elements = 8, 2048

    true_scale = (torch.rand(channels, 1) * 0.4 + 0.05).float()
    codes = torch.randint(qmin, qmax + 1, (channels, elements)).float()
    codes[:, 0] = qmax  # full-scale code present
    codes[:, 1] = 1  # min-magnitude code present (absmin anchor holds)
    input_qdq = (codes * true_scale).to(storage_dtype).float()

    input_q, scale = _decompose_prequantized_tensor(
        input_qdq, qmin, qmax, scale_shape=(channels, 1)
    )

    assert scale.shape == (channels, 1)
    assert torch.all((qmin <= input_q) & (input_q <= qmax))

    # Reconstruction error is bounded by the storage rounding floor, not left at
    # the several-dB loss that min/max fallback would incur.
    recon = input_q * scale
    noise = ((input_qdq - recon) ** 2).mean()
    signal = (input_qdq**2).mean()
    sqnr_db = 10 * torch.log10(signal / noise)
    assert sqnr_db > 40


@pytest.mark.parametrize("storage_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bitwidth", [2, 4])
def test_decomposition_native_low_precision_input(
    storage_dtype: torch.dtype, bitwidth: int
):
    """
    When: The pre-quantized tensor is passed in its native narrow-float dtype
          (bf16/fp16), i.e. not upcast to fp32 first
    Then: Decomposition runs (no dtype-view crash) and the recovered scale
          reconstructs the tensor to a high SQNR
    """
    torch.manual_seed(0)
    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1)
    channels, elements = 8, 2048

    true_scale = (torch.rand(channels, 1) * 0.4 + 0.05).to(storage_dtype)
    codes = torch.randint(qmin, qmax + 1, (channels, elements)).to(storage_dtype)
    codes[:, 0] = qmax
    codes[:, 1] = 1
    input_qdq = codes * true_scale  # stays in storage_dtype

    input_q, scale = _decompose_prequantized_tensor(
        input_qdq, qmin, qmax, scale_shape=(channels, 1)
    )

    assert torch.all((qmin <= input_q) & (input_q <= qmax))
    recon = input_q * scale.to(input_q.dtype)
    ref = input_qdq.float()
    noise = ((ref - recon.float()) ** 2).mean()
    signal = (ref**2).mean()
    sqnr_db = 10 * torch.log10(signal / noise)
    assert sqnr_db > 40


@pytest.mark.parametrize("bitwidth", [2, 4])
def test_decomposition_scale_is_least_squares_optimal(bitwidth: int):
    """
    When: A pre-quantized tensor stored with rounding noise is decomposed
    Then: The recovered per-channel scale is no worse than the closed-form
          least-squares scale for the recovered codes (i.e. it reaches the
          reconstruction-MSE optimum, not merely an anchor-element estimate)
    """
    torch.manual_seed(0)
    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1)
    channels, elements = 8, 4096

    true_scale = (torch.rand(channels, 1) * 0.4 + 0.05).float()
    codes = torch.randint(qmin, qmax + 1, (channels, elements)).float()
    codes[:, 0] = qmax
    codes[:, 1] = 1
    input_qdq = (codes * true_scale).bfloat16().float()

    input_q, scale = _decompose_prequantized_tensor(
        input_qdq, qmin, qmax, scale_shape=(channels, 1)
    )

    # Closed-form LS scale for the recovered codes: <x, q> / <q, q>.
    ls_scale = (input_qdq * input_q).sum(dim=1, keepdim=True) / (
        (input_q**2).sum(dim=1, keepdim=True).clamp(min=1e-12)
    )
    recovered_err = ((input_q * scale - input_qdq) ** 2).sum()
    ls_err = ((input_q * ls_scale - input_qdq) ** 2).sum()
    assert recovered_err <= ls_err * (1 + 1e-4)


def test_decomposition_rejects_unquantized_tensor():
    """
    When: A genuinely unquantized (continuous) tensor is passed
    Then: Decomposition fails so the caller falls back to regular calibration
    """
    torch.manual_seed(0)
    unquantized = torch.randn(8, 2048).bfloat16().float()
    with pytest.raises(_DecompositionError):
        _decompose_prequantized_tensor(unquantized, -8, 8, scale_shape=(8, 1))


def test_decomposition_early_exit_skips_divisor_search():
    """
    When: A large continuous tensor whose blocks carry far more distinct
          magnitudes than the grid could produce is passed
    Then: It is rejected by the early-exit gate, before the divisor search runs
          (i.e. the reject comes from the cheap distinct-magnitude bound, not the
          expensive _select_divisor fallback)
    """
    torch.manual_seed(0)
    # block_numel (2048) > _EARLY_EXIT_LEVEL_SLACK * num_levels (4 * 17 = 68), and
    # ~one distinct magnitude per element, so nearly every block trips the cap.
    unquantized = torch.randn(8, 2048)
    with mock.patch(
        "aimet_torch.utils._select_divisor",
        side_effect=AssertionError("divisor search must not run after early exit"),
    ) as select_divisor:
        with pytest.raises(_DecompositionError, match="early exit"):
            _decompose_prequantized_tensor(unquantized, -8, 8, scale_shape=(8, 1))
    select_divisor.assert_not_called()


def test_decomposition_early_exit_gate_respects_block_size_threshold():
    """
    When: A continuous tensor whose block_numel is at/below
          _EARLY_EXIT_LEVEL_SLACK * num_levels is passed
    Then: The early-exit gate is skipped (too few elements for the distinct-count
          bound to be meaningful) and the divisor search is still consulted before
          any rejection
    """
    qmin, qmax = -8, 8
    num_levels = qmax - qmin + 1
    # block_numel == cap, so `x.shape[1] > slack * num_levels` is False.
    block_numel = aimet_torch.utils._EARLY_EXIT_LEVEL_SLACK * num_levels
    torch.manual_seed(0)
    unquantized = torch.randn(8, block_numel)
    with mock.patch(
        "aimet_torch.utils._select_divisor",
        wraps=aimet_torch.utils._select_divisor,
    ) as select_divisor:
        # Whether it ultimately decomposes or rejects, the point is that the gate
        # did not short-circuit: the divisor search was reached.
        try:
            _decompose_prequantized_tensor(unquantized, qmin, qmax, scale_shape=(8, 1))
        except _DecompositionError as e:
            assert "early exit" not in str(e)
    select_divisor.assert_called_once()


@pytest.mark.parametrize("absmax", [127, 100])
def test_decomposition_recovers_near_constant_channel(absmax: int):
    """
    When: A block's codes are large and consecutive but never reach +/-1 -- e.g.
          weights [absmax - 2, absmax - 1, absmax, ...] whose true per-step scale
          is 1.0 -- so a min-nonzero anchor (which assumes the smallest magnitude
          is one step) would collapse the whole block onto a single code
    Then: The magnitude-gap anchor still finds the true fine grid, so the block is
          decomposed losslessly with scale 1.0 rather than onto the collapsed
          absmax-scale grid (regression guard for near-constant channels, whose
          collapse silently degrades reconstruction SQNR)
    """
    qmin, qmax = -128, 127
    channels, elements = 8, 129
    # Three consecutive magnitudes one step apart; min magnitude == absmax - 2.
    pattern = torch.tensor([absmax - 2, absmax - 1, absmax], dtype=torch.float32)
    weight = pattern.repeat(channels, elements // 3 + 1)[:, :elements].contiguous()

    input_q, scale = _decompose_prequantized_tensor(
        weight, qmin, qmax, scale_shape=(channels, 1)
    )
    assert torch.allclose(scale, torch.ones_like(scale))
    assert torch.allclose(input_q * scale, weight)


def test_decomposition_keeps_genuine_single_level_grid():
    """
    When: A block is genuinely single-valued -- every nonzero element is the exact
          same magnitude (an all-+/-1 pre-quantized channel), so min == absmax
    Then: The single-level grid is accepted (not mistaken for the degenerate case),
          and the recovered scale equals that shared magnitude
    """
    scale = torch.tensor([[0.05], [0.10], [0.20]])
    codes = torch.ones(3, 64)  # every element maps to code +1
    input_qdq = codes * scale

    input_q, recovered = _decompose_prequantized_tensor(
        input_qdq, -8, 7, scale_shape=(3, 1)
    )
    assert torch.allclose(recovered, scale)
    assert torch.all(input_q == 1.0)


def test_decomposition_single_element_blocks():
    """
    When: The scale is per-element, so every block has exactly one element and the
          magnitude-gap estimate has no adjacent pair to difference
    Then: Decomposition runs without a zero-size reduction crash; each block's
          scale is its own magnitude (code +/-1)
    """
    input_qdq = torch.tensor([[5.0], [3.0], [100.0], [0.0]])

    input_q, scale = _decompose_prequantized_tensor(
        input_qdq, -128, 127, scale_shape=(4, 1)
    )
    # Nonzero single-element blocks map to code +/-1 with scale == their magnitude.
    assert torch.allclose(scale.flatten()[:3], torch.tensor([5.0, 3.0, 100.0]))
    assert torch.allclose(input_q.flatten()[:3].abs(), torch.ones(3))
