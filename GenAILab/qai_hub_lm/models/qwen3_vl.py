# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen-3-VL model class"""

import contextlib
import typing
from collections import OrderedDict

import torch

from transformers import AutoConfig, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from GenAILab.qai_hub_lm.models.base import VLM
from GenAILab.qai_hub_lm.models.generator import VLM_Generator
from GenAILab.qai_hub_lm.utils.layer_cache import LayerCacheDescriptor
from GenAILab.qai_hub_lm.utils.model_utils import (
    PositionIdContext,
    compute_vision_input_shapes,
)


class Qwen_3_VL(VLM):
    """Generic quantized Qwen 3 VL"""

    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

    @classmethod
    def instantiate_model(cls, model_id: str, small_model=False) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        )
        if small_model:
            llm_config.num_hidden_layers = 2
            if (
                hasattr(llm_config, "layer_types")
                and llm_config.layer_types is not None
            ):
                llm_config.layer_types = llm_config.layer_types[:2]
        return modeling_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> ProcessorMixin:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID
        return AutoProcessor.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def get_num_deepstack_layers(cls, config):
        """Return the number of deepstack layers from the vision config.

        Falls back to 3 (the Qwen3-VL default) when called with a text-only
        config that lacks ``vision_config``.
        """
        vis_cfg = getattr(config, "vision_config", None)
        if vis_cfg is not None:
            ds_indexes = getattr(vis_cfg, "deepstack_visual_indexes", None)
            if ds_indexes is not None:
                return len(ds_indexes)
        return 3

    @classmethod
    def get_num_visual_tokens(cls, config, image_size=None):
        """Compute the number of visual tokens produced by the vision encoder.

        Returns the **post-merge** token count — the number of tokens that
        appear in the text sequence (and that ``deepstack_visual_embeds``
        contains per layer).  ``compute_vision_input_shapes`` returns the
        pre-merge (vision-model input) count; we divide by
        ``spatial_merge_size²`` to get the merged count.

        This is deterministic given the image size and vision config, and must
        be used consistently at both ONNX export time and inference time so
        that the traced shapes match.
        """
        if image_size is None:
            image_size = (512, 512)
        vis_cfg = getattr(config, "vision_config", None)
        if vis_cfg is None:
            # Text-only config (e.g. from the language model wrapper).
            # Fall back to the Qwen3-VL-4B default: patch=16, merge=2,
            # temporal=2, channels=3 → 1024 pre-merge → 256 post-merge
            # for 512×512.
            return 256
        num_patches, _, _, _ = compute_vision_input_shapes(image_size, vis_cfg)
        merge_size = vis_cfg.spatial_merge_size
        return num_patches // (merge_size * merge_size)

    @classmethod
    def get_sample_backbone_inputs(
        cls,
        model,
        context_length: int,
        sequence_length: int,
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        image_size: tuple[int, int] | None = None,
        config: AutoConfig | None = None,
        *args,
        **kwargs,
    ):
        if config is None:
            config = model.config

        num_visual_tokens = cls.get_num_visual_tokens(config, image_size)
        effective_visual_tokens = min(num_visual_tokens, sequence_length)
        num_deepstack = cls.get_num_deepstack_layers(config)
        hidden_size = model.config.hidden_size

        dummy_inputs_embeds = torch.zeros(
            (1, sequence_length, hidden_size), dtype=torch.int
        )
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)
        dummy_position_ids = torch.zeros((3, 1, sequence_length), dtype=torch.int)

        dummy_visual_pos_masks = torch.zeros((1, sequence_length), dtype=torch.bool)
        start = (sequence_length - effective_visual_tokens) // 2
        end = start + effective_visual_tokens
        dummy_visual_pos_masks[0, start:end] = True

        dummy_deepstack_visual_embeds = [
            torch.zeros(effective_visual_tokens, hidden_size)
            for _ in range(num_deepstack)
        ]

        # Use the same prepare_inputs path as inference so export and
        # runtime shapes are guaranteed to match.
        prepared = Qwen3VL_Generator.prepare_inputs(
            model=model,
            input_ids=None,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
            inputs_embeds=dummy_inputs_embeds,
            position_ids=dummy_position_ids,
            layer_cache_descriptors=layer_cache_descriptors,
            visual_pos_masks=dummy_visual_pos_masks,
            deepstack_visual_embeds=dummy_deepstack_visual_embeds,
        )
        return tuple(prepared.values())

    @classmethod
    def get_sample_vision_inputs(cls, config, image_size=(512, 512)):
        num_patches, pixel_dim, h_patches, w_patches = compute_vision_input_shapes(
            image_size, config.vision_config
        )
        dummy_pixel_values = torch.ones((num_patches, pixel_dim), dtype=torch.float32)
        dummy_grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.int64)
        return (
            dummy_pixel_values,
            dummy_grid_thw,
            torch.zeros((1, 2048), dtype=torch.bool),
        )

    def generate_position_ids(
        self,
        *args,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        num_new_tokens = input_ids.shape[1]
        attention_mask = attention_mask[:, -num_new_tokens:]

        ctx = PositionIdContext(self.config, modeling_qwen3_vl.Qwen3VLModel)
        position_ids, *_ = modeling_qwen3_vl.Qwen3VLModel.get_rope_index(
            ctx, *args, input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return position_ids[..., -num_new_tokens:].to(dtype=torch.int32)

    @classmethod
    def get_backbone_input_names(
        cls,
        layer_cache_descriptors: list[LayerCacheDescriptor] | None = None,
        config=None,
    ) -> tuple[str, ...]:
        num_ds = cls.get_num_deepstack_layers(config) if config else 0
        names = VLM.get_backbone_input_names(layer_cache_descriptors) + (
            "visual_pos_masks",
        )
        if num_ds > 0:
            names += tuple(f"deepstack_visual_embeds_{i}" for i in range(num_ds))
        else:
            names += ("deepstack_visual_embeds",)
        return names

    @staticmethod
    def use_dynamo_export() -> bool:
        return True

    @staticmethod
    def get_visual_input_names() -> tuple[str, ...]:
        return ("pixel_values", "image_grid_thw", "mask")

    @classmethod
    def get_visual_output_names(cls, config=None) -> tuple[str, ...]:
        num_ds = cls.get_num_deepstack_layers(config) if config else 0
        names = ("image_embeddings", "visual_pos_masks")
        if num_ds > 0:
            names += tuple(f"deepstack_visual_embeds_{i}" for i in range(num_ds))
        else:
            names += ("deepstack_visual_embeds",)
        return names

    @staticmethod
    def get_generator_cls() -> type[VLM_Generator]:
        return Qwen3VL_Generator


class Qwen3VLVisualWrapper(torch.nn.Module):
    # Not moving this into shared code since this is pretty model specific
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_outputs = self.visual(
                pixel_values, grid_thw=image_grid_thw, return_dict=True
            )
            return (
                vision_outputs.pooler_output,
                mask,
                vision_outputs.deepstack_features,
            )
        else:
            return None, mask, []


class Qwen3VL_Generator(VLM_Generator):
    # ------------------------------------------------------------------
    # Helpers for image-aware slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _find_image_boundaries(mask_1d: torch.Tensor) -> list[tuple[int, int]]:
        """Return (start, end) index pairs for each contiguous run of True in *mask_1d*."""
        if mask_1d.sum() == 0:
            return []
        changes = torch.diff(
            mask_1d.int(), prepend=torch.tensor([0], device=mask_1d.device)
        )
        starts = (changes == 1).nonzero(as_tuple=True)[0].tolist()
        ends = (changes == -1).nonzero(as_tuple=True)[0].tolist()
        if len(ends) < len(starts):
            ends.append(len(mask_1d))
        return list(zip(starts, ends))

    def _make_dummy_visual_kwargs(
        self, num_visual_tokens, num_deepstack, hidden_size, device, dtype
    ):
        """Build dummy visual extras for a text-only sub-slice."""
        batch_size = 1
        effective_tokens = min(num_visual_tokens, self.sequence_length)
        dummy_mask = torch.zeros(
            (batch_size, self.sequence_length), dtype=torch.bool, device=device
        )
        start = (self.sequence_length - effective_tokens) // 2
        dummy_mask[:, start : start + effective_tokens] = True
        dummy_ds = [
            torch.zeros(effective_tokens, hidden_size, device=device, dtype=dtype)
            for _ in range(num_deepstack)
        ]
        return {"visual_pos_masks": dummy_mask, "deepstack_visual_embeds": dummy_ds}

    def _pad_deepstack_to(
        self,
        sliced_deepstack,
        visual_pos_masks_slice,
        num_visual_tokens,
        num_visual_in_slice,
        hidden_size,
    ):
        """Pad deepstack up to *num_visual_tokens* and flip False→True in mask."""
        pad_count = num_visual_tokens - num_visual_in_slice
        if pad_count <= 0:
            return sliced_deepstack, visual_pos_masks_slice
        false_positions = (~visual_pos_masks_slice).nonzero(as_tuple=True)[1]
        actual_pad = min(pad_count, len(false_positions))
        sliced_deepstack = [
            torch.cat(
                [
                    ds,
                    torch.zeros(
                        actual_pad, hidden_size, device=ds.device, dtype=ds.dtype
                    ),
                ],
                dim=-2,
            )
            for ds in sliced_deepstack
        ]
        visual_pos_masks_slice = visual_pos_masks_slice.clone()
        visual_pos_masks_slice[0, false_positions[:actual_pad]] = True
        return sliced_deepstack, visual_pos_masks_slice

    # ------------------------------------------------------------------

    def slice_inputs_for_inference(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_length: int,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> typing.Generator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None], None, None
    ]:
        deepstack = kwargs.pop("deepstack_visual_embeds", [])
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)

        num_deepstack = (
            len(deepstack)
            if deepstack
            else Qwen_3_VL.get_num_deepstack_layers(self.config)
        )
        num_visual_tokens = Qwen_3_VL.get_num_visual_tokens(
            self.config, self.image_size
        )
        hidden_size = inputs.shape[-1]

        input_length = inputs.shape[1]
        for idx in range(0, input_length, sequence_length)[::-1]:
            idx = input_length - idx
            slice_start = max(0, idx - sequence_length)
            slice_end = idx

            input_slice = inputs[:, slice_start:slice_end]
            attn_slice = attention_mask[:, slice_start:slice_end]
            pos_slice = (
                position_ids[..., slice_start:slice_end]
                if position_ids is not None
                else None
            )

            if visual_pos_masks is None:
                # Text-only: provide dummy visual extras matching traced shapes
                extra_kwargs = self._make_dummy_visual_kwargs(
                    num_visual_tokens,
                    num_deepstack,
                    hidden_size,
                    inputs.device,
                    inputs.dtype,
                )
                yield (
                    input_slice,
                    attn_slice,
                    pos_slice,
                    (kwargs or {}) | extra_kwargs,
                )
                continue

            mask_slice = visual_pos_masks[:, slice_start:slice_end]
            num_visual_in_slice = int(mask_slice.sum().item())
            previously_consumed = int(visual_pos_masks[:, :slice_start].sum().item())

            if num_visual_in_slice <= num_visual_tokens:
                # Single image or fewer tokens — pad and yield one slice
                ds = [
                    d[
                        ...,
                        previously_consumed : previously_consumed + num_visual_in_slice,
                        :,
                    ]
                    for d in deepstack
                ]
                ds, mask_slice = self._pad_deepstack_to(
                    ds,
                    mask_slice,
                    num_visual_tokens,
                    num_visual_in_slice,
                    hidden_size,
                )
                extra_kwargs = {
                    "visual_pos_masks": mask_slice,
                    "deepstack_visual_embeds": ds,
                }
                yield (
                    input_slice,
                    attn_slice,
                    pos_slice,
                    (kwargs or {}) | extra_kwargs,
                )
            else:
                # Multiple images — split into sub-slices at image boundaries
                # so each backbone call sees at most num_visual_tokens.
                image_runs = self._find_image_boundaries(mask_slice[0])

                # Group consecutive image runs so each group has
                # <= num_visual_tokens visual tokens total.
                groups: list[list[tuple[int, int]]] = []
                current_group: list[tuple[int, int]] = []
                current_count = 0
                for run_start, run_end in image_runs:
                    run_len = run_end - run_start
                    if current_count + run_len > num_visual_tokens and current_group:
                        groups.append(current_group)
                        current_group = []
                        current_count = 0
                    current_group.append((run_start, run_end))
                    current_count += run_len
                if current_group:
                    groups.append(current_group)

                # Determine split points: each group extends from the start
                # of its first image run to either the start of the next
                # group's first image run, or the end of the slice.
                ds_offset = previously_consumed
                for g_idx, group in enumerate(groups):
                    sub_start = 0 if g_idx == 0 else groups[g_idx][0][0]
                    sub_end = (
                        groups[g_idx + 1][0][0]
                        if g_idx + 1 < len(groups)
                        else mask_slice.shape[1]
                    )

                    sub_input = input_slice[:, sub_start:sub_end]
                    sub_attn = attn_slice[:, sub_start:sub_end]
                    sub_pos = (
                        pos_slice[..., sub_start:sub_end]
                        if pos_slice is not None
                        else None
                    )
                    sub_mask = mask_slice[:, sub_start:sub_end]
                    num_vis = int(sub_mask.sum().item())

                    sub_ds = [
                        d[..., ds_offset : ds_offset + num_vis, :] for d in deepstack
                    ]
                    ds_offset += num_vis

                    sub_ds, sub_mask = self._pad_deepstack_to(
                        sub_ds,
                        sub_mask,
                        num_visual_tokens,
                        num_vis,
                        hidden_size,
                    )
                    extra_kwargs = {
                        "visual_pos_masks": sub_mask,
                        "deepstack_visual_embeds": sub_ds,
                    }
                    yield (sub_input, sub_attn, sub_pos, (kwargs or {}) | extra_kwargs)

    @classmethod
    def prepare_inputs(
        cls,
        model: torch.nn.Module,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        pad_token: int = 0,
        attention_mask_min: int = -100,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> OrderedDict[str, torch.Tensor]:
        if visual_pos_masks is not None:
            visual_pos_mask_padding_size = sequence_length - visual_pos_masks.shape[1]
            if visual_pos_mask_padding_size > 0:
                visual_pos_masks_padding = torch.zeros(
                    (visual_pos_masks.shape[0], visual_pos_mask_padding_size),
                    dtype=visual_pos_masks.dtype,
                    device=visual_pos_masks.device,
                )
                visual_pos_masks = torch.cat(
                    (visual_pos_masks_padding, visual_pos_masks), dim=-1
                )

        if isinstance(deepstack_visual_embeds, list):
            for i, ds in enumerate(deepstack_visual_embeds):
                kwargs[f"deepstack_visual_embeds_{i}"] = ds

        return super().prepare_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            sequence_length=sequence_length,
            context_length=context_length,
            pad_token=pad_token,
            attention_mask_min=attention_mask_min,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            **kwargs,
        )


#################################  Exportable Vision Attention for ONNX  #################################


class FastExportableQwen3VLVisionAttention(modeling_qwen3_vl.Qwen3VLVisionAttention):
    """
    Vision attention that uses attention masks instead of loop-based splitting.

    This produces a clean ONNX graph while preserving the windowed attention
    behavior defined by cu_seqlens.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from GenAILab.qai_hub_lm.models.qwen2_vl import _create_block_diagonal_mask

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        # Reshape for attention: [1, num_heads, seq_length, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Create block-diagonal attention mask from cu_seqlens
        # This preserves the windowed attention behavior without loops
        attention_mask = _create_block_diagonal_mask(
            cu_seqlens, seq_length, query_states.dtype, query_states.device
        )

        # Use the attention implementation specified in config
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        else:
            attention_interface = modeling_qwen3_vl.eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


def _exportable_deepstack_process(
    self,
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    visual_embeds: torch.Tensor,
) -> torch.Tensor:
    """Export-friendly replacement for ``_deepstack_process``.

    HF's implementation uses boolean indexing (``hidden_states[mask, :]``)
    which creates a data-dependent intermediate shape that
    ``torch.export.draft_export`` cannot trace.  This version uses
    ``cumsum`` + ``gather`` so all shapes remain static.
    """
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

    # mask_int: [batch, seq_len]  —  1 at visual positions, 0 elsewhere
    mask_int = visual_pos_masks.int()

    # cumsum gives 1-based index of each True position within the visual
    # sequence.  False positions get the same value as the preceding True,
    # but we'll zero them out below.
    cumsum = mask_int.cumsum(dim=-1)  # [batch, seq_len]

    # gather_idx: True positions → 1..num_visual, False positions → 0
    # We prepend a zero row to visual_embeds so index 0 fetches zeros.
    gather_idx = cumsum * mask_int  # [batch, seq_len]
    gather_idx = gather_idx.unsqueeze(-1).expand_as(
        hidden_states
    )  # [batch, seq_len, hidden]

    zero_row = torch.zeros(
        1,
        hidden_states.shape[-1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # [num_visual + 1, hidden]  —  row 0 = zeros, rows 1..n = real embeds
    visual_embeds_ext = torch.cat([zero_row, visual_embeds], dim=0)
    visual_embeds_ext = visual_embeds_ext.unsqueeze(0).expand(
        hidden_states.shape[0], -1, -1
    )  # [batch, num_visual + 1, hidden]

    # Gather produces a full [batch, seq_len, hidden] tensor that is zero
    # at non-visual positions and contains the correct embed at visual ones.
    visual_full = torch.gather(visual_embeds_ext, 1, gather_idx)

    return hidden_states + visual_full


@contextlib.contextmanager
def enable_fast_exportable_vision_attention():
    """
    Context manager that temporarily replaces Qwen3VLVisionAttention with
    the exportable version that uses attention masks instead of loop-based splitting.
    Also patches ``_deepstack_process`` to avoid boolean indexing for export.
    """
    original_attn = modeling_qwen3_vl.Qwen3VLVisionAttention
    modeling_qwen3_vl.Qwen3VLVisionAttention = FastExportableQwen3VLVisionAttention

    try:
        yield
    finally:
        modeling_qwen3_vl.Qwen3VLVisionAttention = original_attn
