# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import random
import tempfile
from typing import Callable

import numpy as np

try:
    from aimet_onnx.common.cache import Cache, SerializationProtocolBase
except ImportError:
    from aimet_torch.common.cache import Cache, SerializationProtocolBase


SEED = 18452
random.seed(SEED)
np.random.seed(SEED)


def _assert_equal_default(output, expected):
    assert type(output) == type(expected)
    assert output == expected


def _test_cache(
    fn, protocol: SerializationProtocolBase = None, assert_equal_fn: Callable = None
):
    if not assert_equal_fn:
        assert_equal_fn = _assert_equal_default

    with tempfile.TemporaryDirectory() as cache_dir:
        cache = Cache()

        call_count = 0

        @cache.mark("test", protocol)
        def _fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fn(*args, **kwargs)

        with cache.enable(cache_dir):
            ret = _fn()

        with cache.enable(cache_dir):
            _ret = _fn()

        assert_equal_fn(ret, _ret)
        assert call_count == 1


def test_cache_number():
    _test_cache(lambda: random.random())


def test_cache_list():
    _test_cache(lambda: [random.random() for _ in range(10)])


def test_cache_tuple():
    _test_cache(lambda: tuple(random.random() for _ in range(10)))


def test_cache_none():
    _test_cache(lambda: None)


def test_cache_numpy_array():
    def assert_equal(x, y):
        assert np.array_equal(x, y)

    _test_cache(lambda: np.random.randn(10, 10), assert_equal_fn=assert_equal)
