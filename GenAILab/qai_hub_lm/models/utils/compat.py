# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility patches and shims for HuggingFace transformers."""

import torch


def _patch_sdpa_mask():
    # In transformers >=5.3.0, _preprocess_mask_arguments derives q_length from
    # inputs_embeds.shape[1], which under torch.jit.trace yields a 0-dim tensor
    # instead of a Python int.
    # Fix: convert 0-dim tensors to int.
    try:
        import transformers.masking_utils as _mu

        _orig = _mu.sdpa_mask

        def _patched(batch_size, q_length, *args, **kwargs):
            if isinstance(q_length, torch.Tensor) and q_length.ndim == 0:
                q_length = q_length.item()
            return _orig(batch_size, q_length, *args, **kwargs)

        _mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = _patched
        _mu.sdpa_mask = _patched
    except (ImportError, AttributeError):
        pass


# TODO: Remove this patch once the fix applied in transformers itself.
_patch_sdpa_mask()


class PositionIdContext:
    """Minimal stand-in for ``self`` when calling HF's unbound ``get_rope_index``.

    HF's ``get_rope_index`` is an instance method that accesses ``self.config``
    and may call sibling methods (e.g. ``self.get_vision_position_ids``).  In our
    framework the position-ID computation must work without a real HF model
    instance (e.g. in the ONNX path).  This proxy satisfies the ``self`` contract
    by holding the config and delegating any other attribute lookups to the
    original HF model *class* (bound to this proxy).
    """

    def __init__(self, config, model_cls):
        self.config = config
        self._model_cls = model_cls

    def __getattr__(self, name):
        attr = getattr(self._model_cls, name)
        if callable(attr):
            return attr.__get__(self, type(self))
        return attr
