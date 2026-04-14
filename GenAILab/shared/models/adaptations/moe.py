# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
import contextlib
import torch
import torch.nn.functional as F

from packaging import version
from importlib.metadata import version as get_version
import transformers
import warnings

transformers_version = get_version("transformers")
if version.parse(transformers_version) >= version.parse("4.51.0") and version.parse(
    transformers_version
) < version.parse("5.0.0"):
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
        Qwen3MoeMLP,
    )

    from GenAILab.shared.helpers.yaml_config_parser import YAMLConfigParser
    from transformers import PreTrainedModel

    # ============================================================================
    # Qwen3 MoE adaptation
    # ============================================================================
    class QcQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(
                hidden_states
            )  #  L, hidden_size -> L, num_experts

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)  # L, K

            # Scatter elements without using constant tensor.
            sparse_routing_weights = (router_logits * 0.0).scatter(
                -1, selected_experts, routing_weights
            )  # (L, E)
            final_hidden_states = torch.zeros(
                (batch_size, sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # Qc Adaptation : used as condition term for the op predication
            selected_experts = selected_experts.reshape(-1)
            hidden_states = hidden_states.reshape(
                batch_size, sequence_length, hidden_dim
            )

            """
            QC Adaptation
            1. Loop over all experts
            """
            for expert_idx in range(self.num_experts):
                # Step 1: get the expert at the given index
                expert_layer = self.experts[expert_idx]
                # Step 2: Fetch the hidden states for all the tokens and not just the tokens that correspond to/ attend to this particular expert
                current_hidden_states = hidden_states
                # Step 3: Pass all the token hidden states to expert layer
                expert_out = expert_layer(current_hidden_states)

                # Step 4: Accumulate the expert outputs based on the sparse weights
                """
                For quantization not using op_predication
                If op_predication:
                # Qc Adaptation : Main op predication logic to enable HTP to choose S/ E experts - the graph we export will ideally have all the expert subgraphs in it.
                    # Workaround : Applied unsqueeze/squeeze due to Cast op issue in QuantSim
                final_hidden_states = torch.where((selected_experts==expert_idx).unsqueeze(0).any(dim=1, keepdim=True).squeeze(0),
                                                                    final_hidden_states + expert_out*sparse_routing_weights[:,expert_idx][..., None].to(expert_out.device),
                                                                    final_hidden_states)
                """
                next_states = expert_out * (
                    sparse_routing_weights[:, expert_idx][..., None].to(
                        expert_out.device
                    )
                )
                final_hidden_states += next_states[0]

            return final_hidden_states, router_logits

    @contextlib.contextmanager
    def qwen3_moe_expert_selection():
        """Context manager to temporarily replace Qwen3Attention with SHA version."""
        original = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
        modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = QcQwen3MoeSparseMoeBlock

        yield

        modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = original

    class Qwen3MoEMixin:
        """Mixin to enable MoE expert selection for Qwen3 models."""

        @classmethod
        def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
            with qwen3_moe_expert_selection():
                model = super().instantiate_model(*args, **kwargs)
            return model

    class Qwen3MoEAdaptation(Qwen3MoEMixin):
        """Expert selection adaptation for Qwen3 MoE models."""

        pass

    YAMLConfigParser.register_adaptation("Expert_Selection", model_type="qwen3_moe")(
        Qwen3MoEAdaptation
    )

else:
    warnings.warn(
        f"Transformers version {transformers.__version__} not supported for Qwen3MoE. "
        f"Only versions >=4.51.0 and <5.0.0 are supported."
    )
    Qwen3MoEAdaptation = None
