# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utils for computing RoPE embeddings outside of the model."""

import typing
import torch


@typing.runtime_checkable
class RopeEmbeddingProtocol(typing.Protocol):
    """Protocol for any object that can provide RoPE (cos, sin) embeddings."""

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class RopeEmbedding:
    def __init__(self, model, context_length):
        embedding_cls = self._lookup_embedding_cls(model.config)
        embedding = embedding_cls(model.config)
        dummy_x = torch.tensor([1.0])
        position_ids = torch.arange(context_length).view(1, -1)
        embeddings = embedding.forward(dummy_x, position_ids)

        # for adapted models
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        self.cos, self.sin = [emb.unsqueeze(0) for emb in embeddings]

    @staticmethod
    def _lookup_embedding_cls(config):
        try:
            from transformers.models.qwen2 import Qwen2Config
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

            if isinstance(config, Qwen2Config):
                return Qwen2RotaryEmbedding
        except ImportError:
            pass

        try:
            from transformers.models.qwen3 import Qwen3Config
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

            if isinstance(config, Qwen3Config):
                return Qwen3RotaryEmbedding
        except ImportError:
            pass

        try:
            from transformers.models.llama import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

            if isinstance(config, LlamaConfig):
                return LlamaRotaryEmbedding
        except ImportError:
            pass

        try:
            from transformers.models.phi import PhiConfig
            from transformers.models.phi.modeling_phi import PhiRotaryEmbedding

            if isinstance(config, PhiConfig):
                return PhiRotaryEmbedding
        except ImportError:
            pass

        raise RuntimeError("Unknown rotary embedding type for model")

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin
