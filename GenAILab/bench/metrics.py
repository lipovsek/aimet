# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Metrics for GenAI testing"""

import gc
import time
import warnings
import yaml
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, GenerationConfig, TextStreamer
from transformers.processing_utils import ProcessorMixin

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from .datasets import (
    Wikitext,
    TinyMMLU as TinyMMLUDataset,
    MMLU as MMLUDataset,
    MMMLU as MMMLUDataset,
    MMMU as MMMUDataset,
)


class EvaluationMetric(ABC):
    pass


class TextEvaluationMetric(EvaluationMetric):
    """Generic GenAI text evaluation metric"""

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext,
    ) -> float | list[str]:
        """Perform evaluation on provided model"""


class DistanceMetric(EvaluationMetric):
    """Base class for metrics that compare quantized model outputs against FP baseline.

    This class is **modality-agnostic**.  Concrete distance metrics opt into a
    modality by also inheriting the appropriate evaluation base:

    *   Text-only:       ``class MMLUKLDiv(DistanceMetric, TextEvaluationMetric)``
        — receives the unwrapped tokenizer on VLMs.
    *   Multimodal:      ``class MMMUFlips(DistanceMetric)``
        — receives the full processor.

    Subclasses use the :class:`EvaluationContext` passed via the ``eval_ctx``
    keyword argument to cache and share intermediate results (e.g. logits)
    across multiple distance metrics without redundant forward passes.

    FP results are persisted to disk and shared across quantization recipes
    and pytest sessions.  Quant results are cached in-memory for the duration
    of a single test.
    """

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext,
        **kwargs,
    ) -> float:
        """Compute a distance metric between quantized and FP model outputs."""


@YAMLConfigParser.register_metric
class PPL(TextEvaluationMetric):
    """PPL evaluation metric"""

    @staticmethod
    def _compute_loss_from_logits(
        output_logits: torch.Tensor, input_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Helper function to compute loss"""

        # Get the outputs and move it to CPU. Assumes that index 0 is logits as
        lm_logits = output_logits.cpu()

        # Trim the last logit off lm_logits, and the first token off input_tokens
        shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        shift_labels = input_tokens[..., 1:].contiguous().to(shift_logits.device)

        loss_fn = torch.nn.CrossEntropyLoss()
        neg_log_likelihood = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return neg_log_likelihood

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        batch_size: int = 1,
        num_iterations: int = None,
    ) -> float:
        dataset = Wikitext.load_encoded_dataset(tokenizer, context_length, "test")
        dataloader = DataLoader(dataset, batch_size=batch_size)

        neg_log_likelihoods = []
        for i, batch in tqdm(
            enumerate(dataloader),
            total=num_iterations or len(dataloader),
            desc="Evaluating PPL",
        ):
            if num_iterations is not None and i >= num_iterations:
                break

            batch["input_ids"] = batch["input_ids"].to(model.device)
            outputs = model(input_ids=batch["input_ids"][0])
            neg_log_likelihoods.append(
                cls._compute_loss_from_logits(outputs[0], batch["input_ids"])
            )
            del outputs

        ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
        return float(ppl)


class GenericMMLU(TextEvaluationMetric):
    """Generic MMLU evaluation metric. Should work with any MMLU dataset."""

    @classmethod
    def get_collection_name(cls):
        """Get the collection name. Used for indexing into the EvaluationContext."""
        return f"{cls.__name__}_choice_logits"

    @staticmethod
    @abstractmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer, context_length: int
    ) -> DataLoader:
        """Get the dataloader associated with this MMLU evaluator."""

    @classmethod
    def collect_choice_logits(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        **kwargs,
    ) -> dict:
        """Run the model over this MMLU variant and collect per-sample data.

        Returns a dict with:

        * ``"logits"`` – ``Tensor(N, 4)`` of raw logits at the A/B/C/D token
          positions (before any softmax).
        * ``"labels"`` – ``Tensor(N,)`` of correct-answer indices (0–3).
        """
        kwargs.pop("image_size", None)
        dataloader = cls.get_dataloader(tokenizer, context_length, **kwargs)

        def tokenize_letter(letter: str):
            return torch.Tensor(
                tokenizer(letter, add_special_tokens=False)["input_ids"]
            ).to(dtype=torch.int)

        choices = tuple(tokenize_letter(letter) for letter in ("A", "B", "C", "D"))

        all_logits = []
        all_labels = []

        for batch in tqdm(
            dataloader, total=len(dataloader), desc=f"Collecting {cls.__name__} logits"
        ):
            batch["input_ids"] = (
                torch.Tensor(batch["input_ids"])
                .to(dtype=torch.int, device=model.device)
                .unsqueeze(0)
            )
            outputs = model(input_ids=batch["input_ids"])

            last_logit = (
                outputs[0][..., -1, :]
                .contiguous()
                .to(dtype=torch.float32, device="cpu")
                .flatten()
            )

            choice_logits = torch.tensor([last_logit[c].item() for c in choices])
            all_logits.append(choice_logits)

            label_token = torch.Tensor(batch["label"]).to(dtype=torch.int)
            label_idx = next(
                i for i, c in enumerate(choices) if torch.equal(c, label_token)
            )
            all_labels.append(label_idx)

            del outputs

        return {
            "logits": torch.stack(all_logits),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> float:
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        def collect_qt():
            return cls.collect_choice_logits(model, tokenizer, context_length, **kwargs)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect_qt)
            if eval_ctx
            else collect_qt()
        )
        preds = data["logits"].argmax(dim=-1)
        correct = (preds == data["labels"]).sum().item()
        return float(correct / len(data["labels"])) * 100


@YAMLConfigParser.register_metric
class TinyMMLU(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer, context_length: int
    ) -> DataLoader:
        dataset = TinyMMLUDataset.load_encoded_dataset(
            tokenizer, context_length, "test"
        )
        return DataLoader(dataset)


@YAMLConfigParser.register_metric
class MMLU(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        num_fewshot: int = 5,
    ) -> DataLoader:
        dataset = MMLUDataset.load_encoded_dataset(
            tokenizer, context_length, "test", num_fewshot=num_fewshot
        )
        return DataLoader(dataset)


@YAMLConfigParser.register_metric
class MMLU1000(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        num_fewshot: int = 5,
    ) -> DataLoader:
        dataset = MMLUDataset.load_encoded_dataset(
            tokenizer, context_length, "test", num_fewshot=num_fewshot
        )
        return DataLoader(Subset(dataset, torch.arange(1000)))


@YAMLConfigParser.register_metric
class MMMLU(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str,
        num_fewshot: int = 5,
    ) -> DataLoader:
        dataset = MMMLUDataset.load_encoded_dataset(
            tokenizer, context_length, split, num_fewshot
        )
        return DataLoader(dataset)


# ---------------------------------------------------------------------------
# MMLU distance metrics
# ---------------------------------------------------------------------------


class _MMLUDistanceBase(DistanceMetric, TextEvaluationMetric):
    """Shared MMLU data collection for all MMLU-based distance metrics.

    Subclasses only need to implement :meth:`_compute`.  The underlying MMLU
    forward passes (both FP and quantized) are run at most once and cached via
    the :class:`EvaluationContext`.  Data collection is delegated to
    :meth:`MMLU.collect_choice_logits` so the iteration logic lives in one
    place, and the quant collection is shared with the :class:`MMLU` accuracy
    metric when both appear in the same test config.
    """

    @classmethod
    def _get_mmlu_data(cls, model, tokenizer, context_length, eval_ctx, num_fewshot=5):
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        # Use the same collection name as MMLU.evaluate so quant results are shared.
        collection = MMLU.get_collection_name()

        def collect_fp():
            with model.fp_mode():
                return MMLU.collect_choice_logits(
                    model, tokenizer, context_length, num_fewshot=num_fewshot
                )

        def collect_qt():
            return MMLU.collect_choice_logits(
                model, tokenizer, context_length, num_fewshot=num_fewshot
            )

        fp = (
            eval_ctx.get_or_compute_fp(collection, collect_fp)
            if eval_ctx
            else collect_fp()
        )
        q = (
            eval_ctx.get_or_compute_quant(collection, collect_qt)
            if eval_ctx
            else collect_qt()
        )
        return fp, q

    @classmethod
    @abstractmethod
    def _compute(cls, fp_data: dict, q_data: dict) -> float:
        """Compute the metric from collected FP and quantized MMLU data."""

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        num_fewshot: int = 5,
        **kwargs,
    ):
        fp, q = cls._get_mmlu_data(
            model, tokenizer, context_length, eval_ctx, num_fewshot
        )
        return cls._compute(fp, q)


# ---------------------------------------------------------------------------
# Shared distance-metric computation mixins
# ---------------------------------------------------------------------------


class _KLDivergenceCompute:
    """KL divergence KL(P_fp || P_quant) over answer choice distributions."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        p = torch.nn.functional.softmax(fp_data["logits"], dim=-1)
        log_q = torch.nn.functional.log_softmax(q_data["logits"], dim=-1)
        return torch.nn.functional.kl_div(log_q, p, reduction="batchmean").item()


class _ReverseKLDivergenceCompute:
    """Reverse KL divergence KL(P_quant || P_fp) over answer choice distributions."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        q = torch.nn.functional.softmax(q_data["logits"], dim=-1)
        log_p = torch.nn.functional.log_softmax(fp_data["logits"], dim=-1)
        return torch.nn.functional.kl_div(log_p, q, reduction="batchmean").item()


class _FlipsCompute:
    """Percentage of samples where quantized and FP predictions disagree."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        fp_preds = fp_data.get("preds", fp_data["logits"].argmax(dim=-1))
        q_preds = q_data.get("preds", q_data["logits"].argmax(dim=-1))
        return (fp_preds != q_preds).float().mean().item() * 100


class _JSDivergenceCompute:
    """Jensen-Shannon divergence between FP and quantized distributions."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        p = torch.nn.functional.softmax(fp_data["logits"], dim=-1)
        q = torch.nn.functional.softmax(q_data["logits"], dim=-1)
        m = 0.5 * (p + q)
        kl_pm = torch.nn.functional.kl_div(m.log(), p, reduction="batchmean")
        kl_qm = torch.nn.functional.kl_div(m.log(), q, reduction="batchmean")
        return (0.5 * (kl_pm + kl_qm)).item()


@YAMLConfigParser.register_metric
class MMLUKLDivergence(_KLDivergenceCompute, _MMLUDistanceBase):
    """KL divergence KL(P_fp || P_quant) over MMLU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMLUReverseKLDivergence(_ReverseKLDivergenceCompute, _MMLUDistanceBase):
    """Reverse KL divergence KL(P_quant || P_fp) over MMLU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMLUFlips(_FlipsCompute, _MMLUDistanceBase):
    """Percentage of MMLU samples where quantized and FP predictions disagree."""


@YAMLConfigParser.register_metric
class MMLUJSDivergence(_JSDivergenceCompute, _MMLUDistanceBase):
    """Jensen-Shannon divergence between FP and quantized MMLU distributions."""


# ---------------------------------------------------------------------------
# MMMU metrics (multimodal)
# ---------------------------------------------------------------------------


@YAMLConfigParser.register_metric
class MMMU(EvaluationMetric):
    """Generic MMMU evaluation metric for multimodal models."""

    @classmethod
    def get_collection_name(cls):
        """Get the collection name. Used for indexing into the EvaluationContext."""
        return f"{cls.__name__}_choice_logits"

    @staticmethod
    def get_dataset(processor, context_length, image_size=None, **kwargs):
        return MMMUDataset.load_encoded_dataset(
            processor, context_length, split="validation", image_size=image_size
        )

    @classmethod
    def collect_choice_logits(cls, model, processor, context_length, **kwargs) -> dict:
        """Run the model over MMMU and collect per-sample choice logits.

        Returns a dict with:

        * ``"logits"`` – ``Tensor(N, max_options)`` of raw logits at each
          answer-letter token position, padded with ``-inf`` for samples that
          have fewer options than the maximum.
        * ``"preds"``  – ``Tensor(N,)`` of predicted answer indices.
        * ``"labels"`` – ``Tensor(N,)`` of correct-answer indices.
        """
        dataset = cls.get_dataset(processor, context_length, **kwargs)

        tokenizer = getattr(processor, "tokenizer", processor)

        def _token_id(letter):
            return tokenizer(letter, add_special_tokens=False)["input_ids"][0]

        all_logits = []  # variable-length per sample, padded later
        all_preds = []
        all_labels = []

        for sample in tqdm(dataset, desc=f"Collecting {cls.__name__} logits"):
            num_options = sample.pop("num_options", 4)
            label = sample.pop("label")

            inputs = {
                k: v.to(model.device)
                for k, v in sample.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = model(**inputs)

            last_logit = (
                outputs[0][..., -1, :]
                .contiguous()
                .to(dtype=torch.float32, device="cpu")
                .flatten()
            )

            # Only compare logits for the actual number of options
            choice_letters = [chr(65 + i) for i in range(num_options)]
            choice_ids = [_token_id(c) for c in choice_letters]
            choice_logits = torch.tensor([last_logit[c].item() for c in choice_ids])

            all_logits.append(choice_logits)
            all_preds.append(choice_logits.argmax().item())
            all_labels.append(ord(label.strip().upper()) - ord("A"))

            del outputs, inputs
            torch.cuda.empty_cache()

        # Pad logits to the maximum number of options with -inf so they can be
        # stacked into a single tensor.  Softmax(-inf) == 0 so padded positions
        # contribute nothing to KL / JS divergence computations.
        max_options = max(l.size(0) for l in all_logits) if all_logits else 4
        padded = []
        for logit in all_logits:
            pad_len = max_options - logit.size(0)
            if pad_len > 0:
                logit = torch.cat([logit, logit.new_full((pad_len,), float("-inf"))])
            padded.append(logit)

        return {
            "logits": torch.stack(padded),
            "preds": torch.tensor(all_preds, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> float:
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        def collect_qt():
            return cls.collect_choice_logits(model, processor, context_length, **kwargs)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect_qt)
            if eval_ctx
            else collect_qt()
        )
        correct = (data["preds"] == data["labels"]).sum().item()
        return float(correct / len(data["labels"])) * 100


# ---------------------------------------------------------------------------
# MMMU distance metrics
# ---------------------------------------------------------------------------


class _MMMUDistanceBase(DistanceMetric):
    """Shared MMMU data collection for all MMMU-based distance metrics"""

    @classmethod
    def _get_mmmu_data(
        cls, model, processor, context_length, eval_ctx, image_size=None
    ):
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        collection = MMMU.get_collection_name()

        def collect_fp():
            with model.fp_mode():
                return MMMU.collect_choice_logits(
                    model, processor, context_length, image_size=image_size
                )

        def collect_qt():
            return MMMU.collect_choice_logits(
                model, processor, context_length, image_size=image_size
            )

        fp = (
            eval_ctx.get_or_compute_fp(collection, collect_fp)
            if eval_ctx
            else collect_fp()
        )
        q = (
            eval_ctx.get_or_compute_quant(collection, collect_qt)
            if eval_ctx
            else collect_qt()
        )
        return fp, q

    @classmethod
    @abstractmethod
    def _compute(cls, fp_data: dict, q_data: dict) -> float:
        """Compute the metric from collected FP and quantized MMMU data."""

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        num_fewshot: int = 5,
        image_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        fp, q = cls._get_mmmu_data(
            model, processor, context_length, eval_ctx, image_size=image_size
        )
        return cls._compute(fp, q)


@YAMLConfigParser.register_metric
class MMMUKLDivergence(_KLDivergenceCompute, _MMMUDistanceBase):
    """KL divergence KL(P_fp || P_quant) over MMMU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMMUReverseKLDivergence(_ReverseKLDivergenceCompute, _MMMUDistanceBase):
    """Reverse KL divergence KL(P_quant || P_fp) over MMMU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMMUFlips(_FlipsCompute, _MMMUDistanceBase):
    """Percentage of MMMU samples where quantized and FP predictions disagree."""


@YAMLConfigParser.register_metric
class MMMUJSDivergence(_JSDivergenceCompute, _MMMUDistanceBase):
    """Jensen-Shannon divergence between FP and quantized MMMU distributions."""


class TimedStreamer(TextStreamer):
    """TextStreamer that records prefill and decode timing stats."""

    def __init__(self, *args, num_input_tokens: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = 0
        self.first_token_time = None
        self.end_time = None
        self.start_time = time.perf_counter()

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.num_output_tokens += value.numel()
        super().put(value)

    def end(self):
        self.end_time = time.perf_counter()
        super().end()
        self._print_stats()

    def _print_stats(self):
        ttft = (
            self.first_token_time - self.start_time if self.first_token_time else None
        )
        decode_tokens = max(self.num_output_tokens - 1, 0)
        decode_time = (
            (self.end_time - self.first_token_time) if self.first_token_time else 0
        )

        print(f"\n--- Generation Stats ---")
        print(f"  Input tokens:  {self.num_input_tokens}")
        print(f"  Output tokens: {self.num_output_tokens}")
        if ttft is not None:
            print(
                f"  TTFT:          {ttft:.3f}s  ({self.num_input_tokens / ttft:.1f} prefill tok/s)"
            )
        if decode_time > 0 and decode_tokens > 0:
            print(
                f"  Decode:        {decode_time:.3f}s  ({decode_tokens / decode_time:.1f} tok/s)"
            )
        print(f"  Total:         {self.end_time - self.start_time:.3f}s")


@YAMLConfigParser.register_metric
class Interactive(TextEvaluationMetric):
    @staticmethod
    def _get_generation_config(model, tokenizer, **overrides) -> GenerationConfig:
        """Build a GenerationConfig with EOS tokens merged from model config and tokenizer."""
        eos_ids = set()
        for src in (
            getattr(model.config, "eos_token_id", None),
            tokenizer.eos_token_id,
        ):
            if src is None:
                continue
            if isinstance(src, (list, tuple)):
                eos_ids.update(src)
            else:
                eos_ids.add(src)

        defaults = dict(
            max_length=2048,
            eos_token_id=sorted(eos_ids) if eos_ids else tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            temperature=0.8,
        )
        defaults.update(overrides)
        return GenerationConfig(**defaults)

    @staticmethod
    def get_system_prompt() -> str:
        return "You are a helpful AI assistant."

    @classmethod
    def generate_output(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        unformatted_prompt: str = None,
        formatted_prompt: str = None,
        generation_config: GenerationConfig = None,
        highlight_output: bool = False,
    ) -> str:
        if formatted_prompt is None and unformatted_prompt is None:
            raise ValueError(
                "Either unformatted_prompt or formatted_prompt must be provided."
            )
        if formatted_prompt is not None and unformatted_prompt is not None:
            raise ValueError(
                "Only one of unformatted_prompt or formatted_prompt should be provided."
            )

        if formatted_prompt is None:
            formatted_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": cls.get_system_prompt()},
                    {"role": "user", "content": unformatted_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        tokenized_user_input = tokenizer(formatted_prompt, return_tensors="pt").to(
            model.device
        )

        model.generation_config = (
            generation_config
            if generation_config is not None
            else cls._get_generation_config(model, tokenizer)
        )

        print(formatted_prompt, end="")
        if highlight_output:
            print("\033[0;31m", end="")  # Start red color for output

        streamer = TimedStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            num_input_tokens=tokenized_user_input["input_ids"].shape[-1],
        )
        outputs = model.generate(
            inputs=tokenized_user_input["input_ids"],
            attention_mask=tokenized_user_input["attention_mask"],
            generation_config=model.generation_config,
            streamer=streamer,
        )

        if highlight_output:
            print("\033[0m")  # Reset color after highlighted output

        # Detokenize and return the generated string
        generated_tokens = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return generated_text

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
    ) -> float:
        while True:
            user_input_prompt = input("Enter your prompt or 'exit' to quit: ")
            if user_input_prompt == "exit":
                break
            cls.generate_output(model, tokenizer, unformatted_prompt=user_input_prompt)
        return float("nan")


@YAMLConfigParser.register_metric
class TrickyPrompts(Interactive):
    prompts = {
        "phi3": [
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\nWhat is Gravity?<|end|>\n<|assistant|>\nGravity is a fundamental force of nature that attracts two bodies with mass towards each other. It is described by Isaac Newton'",
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\nWhat is Gravity?<|end|>\n<|assistant|>\nGravity is a fundamental force of nature that attracts two bodies with mass towards each other. It is described by Isaac Newton's theory in the 17th century and is a key component in Albert Einstein'",
        ]
    }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
    ) -> list[str]:
        generated_text = []
        for prompt in TrickyPrompts.prompts.get(model.config.model_type, []):
            print("===============================")
            generated_text.append(
                cls.generate_output(
                    model,
                    tokenizer,
                    formatted_prompt=prompt,
                    generation_config=cls._get_generation_config(
                        model,
                        tokenizer,
                        max_new_tokens=2,
                        do_sample=False,
                    ),
                    highlight_output=True,
                )
            )
        print("===============================")
        return generated_text


@YAMLConfigParser.register_metric
class Prompts(Interactive):
    PROMPTS_FILE = Path(__file__).parent / "prompts" / "text_prompts.yaml"

    @classmethod
    def get_collection_name(cls):
        return f"{cls.__name__}_generated_text"

    @classmethod
    def _load_prompts(cls):
        with open(cls.PROMPTS_FILE) as f:
            return yaml.safe_load(f)

    @classmethod
    def _normalize_prompt(cls, entry) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            k, v = next(iter(entry.items()))
            return f"{k}: {v}"
        return str(entry)

    @classmethod
    def _generate_all(cls, model, tokenizer):
        raw_prompts = cls._load_prompts()
        prompts = [cls._normalize_prompt(p) for p in raw_prompts]
        generated_text = []
        for prompt in prompts:
            print("===============================")
            generated_text.append(
                cls.generate_output(
                    model=model,
                    tokenizer=tokenizer,
                    unformatted_prompt=prompt,
                    generation_config=cls._get_generation_config(
                        model, tokenizer, do_sample=False
                    ),
                )
            )
        print("===============================")
        return {"prompts": prompts, "generated_text": generated_text}

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
    ) -> list[str]:
        def collect():
            return cls._generate_all(model, tokenizer)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect)
        else:
            data = collect()

        return data["generated_text"]


@YAMLConfigParser.register_metric
class MultimodalPrompts(EvaluationMetric):
    PROMPTS_FILE = Path(__file__).parent / "prompts" / "multimodal_prompts.yaml"
    IMAGE_DIR = Path(__file__).parent / "prompts" / "sample_images"

    @classmethod
    def get_collection_name(cls):
        return f"{cls.__name__}_generated_text"

    @classmethod
    def _load_prompts(cls):
        with open(cls.PROMPTS_FILE) as f:
            return yaml.safe_load(f)

    @classmethod
    def _generate_all(cls, model, processor):
        from PIL import Image

        if model.generation_config is None:
            model.generation_config = GenerationConfig()

        tokenizer = getattr(processor, "tokenizer", processor)
        prompts = cls._load_prompts()
        generated_text = []

        for entry in prompts:
            image_file = entry["image"]
            prompt_text = entry["prompt"]
            print("===============================")
            image_path = cls.IMAGE_DIR / image_file
            image = Image.open(image_path).convert("RGB")
            if model.image_size is not None:
                image = image.resize(model.image_size)

            content = [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            inputs.pop("mm_token_type_ids", None)

            generation_config = Interactive._get_generation_config(
                model,
                tokenizer,
                do_sample=False,
            )

            streamer = TimedStreamer(
                tokenizer=tokenizer,
                skip_prompt=True,
                num_input_tokens=inputs["input_ids"].shape[-1],
            )
            print(text, end="")
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                streamer=streamer,
            )

            generated_tokens = (
                outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            )
            result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            generated_text.append(result)

        print("===============================")
        return {"prompts": prompts, "generated_text": generated_text}

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> list[str]:
        if not isinstance(model, VLM_Generator):
            raise ValueError("MultimodalPrompts metric requires a VL model.")

        def collect():
            return cls._generate_all(model, processor)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect)
        else:
            data = collect()

        return data["generated_text"]


@YAMLConfigParser.register_metric
class AutogradedPrompts(TextEvaluationMetric):
    """Grade generated responses with a small LLM as a 4-way classifier (A/B/C/D).

    For each prompt/response pair, a single forward pass is run through the
    grader model. The logits at the last token position are read and argmax is
    taken over the four letter-token IDs. Points are assigned per the harness
    config and the final score is reported as a percentage of max possible.
    """

    HARNESS_FILE = (
        Path(__file__).parent / "prompts" / "text_prompts_autograder_harness.yaml"
    )
    LETTERS = ("A", "B", "C", "D")
    DEFAULT_HARNESS_VERSION = "v1"

    @classmethod
    def _load_harness(cls, version: str = DEFAULT_HARNESS_VERSION):
        with open(cls.HARNESS_FILE) as f:
            harness = yaml.safe_load(f)
        return harness[version]

    @staticmethod
    def _get_letter_token_ids(tokenizer) -> list[int]:
        ids = []
        for letter in AutogradedPrompts.LETTERS:
            tok_ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                tok_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                raise ValueError(
                    f"Letter {letter!r} tokenizes to {len(tok_ids)} tokens; "
                    f"grader needs single-token letters."
                )
            ids.append(tok_ids[0])
        if len(set(ids)) != 4:
            raise ValueError(f"Letter token ids collided: {ids}")
        return ids

    @classmethod
    def _score_one(
        cls,
        grader_model,
        grader_tokenizer,
        grading_prompt,
        prompt,
        response,
        letter_ids,
    ) -> str:
        text = grading_prompt.replace("{prompt}", prompt).replace(
            "{response}", response
        )
        messages = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": ""},
        ]
        formatted = grader_tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True, enable_thinking=False
        )
        input_ids = grader_tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(grader_model.device)
        outputs = grader_model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :].float().cpu()
        choice_logits = {
            letter: logits[tok_id].item()
            for letter, tok_id in zip(cls.LETTERS, letter_ids)
        }
        return max(choice_logits, key=choice_logits.get)

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        harness_version: str = DEFAULT_HARNESS_VERSION,
        **kwargs,
    ) -> float:
        def collect():
            return Prompts._generate_all(model, tokenizer)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(Prompts.get_collection_name(), collect)
        else:
            data = collect()

        prompts = data["prompts"]
        responses = data["generated_text"]

        harness = cls._load_harness(harness_version)
        grader_model_id = harness["model_id"]
        grading_prompt = harness["grading_prompt"]
        letter_points = harness["letter_points"]
        max_points = max(letter_points.values())

        from transformers import AutoModelForCausalLM, AutoTokenizer

        with model.on_device(torch.device("cpu")):
            grader_tokenizer = AutoTokenizer.from_pretrained(grader_model_id)
            grader_model = AutoModelForCausalLM.from_pretrained(
                grader_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            grader_model.eval()
            letter_ids = cls._get_letter_token_ids(grader_tokenizer)

            total_points = 0
            for prompt, response in tqdm(
                zip(prompts, responses),
                total=len(prompts),
                desc="Autograding responses",
            ):
                pred = cls._score_one(
                    grader_model,
                    grader_tokenizer,
                    grading_prompt,
                    prompt,
                    response,
                    letter_ids,
                )
                total_points += letter_points[pred]

            del grader_model
            del grader_tokenizer

            gc.collect()
            torch.cuda.empty_cache()

        return 100.0 * total_points / (max_points * len(prompts))


@YAMLConfigParser.register_metric
class AutogradedMultimodalPrompts(EvaluationMetric):
    """Grade VLM responses with an external VLM as a 4-way classifier (A/B/C/D).

    The grader model receives both the source image and the generated response.
    Scoring uses single-forward-pass argmax over letter tokens, same as
    AutogradedPrompts.
    """

    HARNESS_FILE = (
        Path(__file__).parent / "prompts" / "multimodal_prompts_autograder_harness.yaml"
    )
    IMAGE_DIR = Path(__file__).parent / "prompts" / "sample_images"
    LETTERS = ("A", "B", "C", "D")
    DEFAULT_HARNESS_VERSION = "v1"

    @classmethod
    def _load_harness(cls, version: str = DEFAULT_HARNESS_VERSION):
        with open(cls.HARNESS_FILE) as f:
            harness = yaml.safe_load(f)
        return harness[version]

    @staticmethod
    def _get_letter_token_ids(tokenizer) -> list[int]:
        ids = []
        for letter in AutogradedMultimodalPrompts.LETTERS:
            tok_ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                tok_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                raise ValueError(
                    f"Letter {letter!r} tokenizes to {len(tok_ids)} tokens; "
                    f"grader needs single-token letters."
                )
            ids.append(tok_ids[0])
        if len(set(ids)) != 4:
            raise ValueError(f"Letter token ids collided: {ids}")
        return ids

    @classmethod
    def _score_one(
        cls,
        grader_model,
        grader_processor,
        grading_prompt,
        prompt_text,
        response,
        image,
        letter_ids,
    ):
        text = grading_prompt.replace("{prompt}", prompt_text).replace(
            "{response}", response
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            },
            {"role": "assistant", "content": ""},
        ]
        formatted = grader_processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True, enable_thinking=False
        )
        inputs = grader_processor(text=[formatted], images=[image], return_tensors="pt")
        inputs = {k: v.to(grader_model.device) for k, v in inputs.items()}

        outputs = grader_model(**inputs)
        logits = outputs.logits[0, -1, :].float().cpu()
        choice_logits = {
            letter: logits[tok_id].item()
            for letter, tok_id in zip(cls.LETTERS, letter_ids)
        }
        return max(choice_logits, key=choice_logits.get)

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        harness_version: str = DEFAULT_HARNESS_VERSION,
        **kwargs,
    ) -> float:
        if not isinstance(model, VLM_Generator):
            raise ValueError("AutogradedMultimodalPrompts requires a VL model.")

        def collect():
            return MultimodalPrompts._generate_all(model, processor)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(
                MultimodalPrompts.get_collection_name(), collect
            )
        else:
            data = collect()

        prompts = data["prompts"]
        responses = data["generated_text"]

        harness = cls._load_harness(harness_version)
        grader_model_id = harness["model_id"]
        grading_prompt = harness["grading_prompt"]
        letter_points = harness["letter_points"]
        max_points = max(letter_points.values())

        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        with model.on_device(torch.device("cpu")):
            grader_processor = AutoProcessor.from_pretrained(grader_model_id)
            grader_model = AutoModelForImageTextToText.from_pretrained(
                grader_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            grader_model.eval()
            grader_tokenizer = getattr(grader_processor, "tokenizer", grader_processor)
            letter_ids = cls._get_letter_token_ids(grader_tokenizer)

            total_points = 0
            for entry, response in tqdm(
                zip(prompts, responses),
                total=len(prompts),
                desc="Autograding multimodal responses",
            ):
                image_path = cls.IMAGE_DIR / entry["image"]
                image = Image.open(image_path).convert("RGB")

                pred = cls._score_one(
                    grader_model,
                    grader_processor,
                    grading_prompt,
                    entry["prompt"],
                    response,
                    image,
                    letter_ids,
                )
                total_points += letter_points[pred]

            del grader_model
            del grader_processor

            gc.collect()
            torch.cuda.empty_cache()

        return 100.0 * total_points / (max_points * len(prompts))
