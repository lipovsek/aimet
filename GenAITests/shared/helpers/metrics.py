# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Metrics for GenAI testing"""

import warnings
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, GenerationConfig, TextStreamer
from transformers.processing_utils import ProcessorMixin

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.helpers.eval_context import EvaluationContext
from GenAITests.shared.models.generator import Generator
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


@YAMLConfigParser.register_metric
class Interactive(TextEvaluationMetric):
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
            else GenerationConfig(
                max_new_tokens=1000,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=40,
                top_p=0.95,
                temperature=0.8,
            )
        )

        print(formatted_prompt, end="")
        if highlight_output:
            print("\033[0;31m", end="")  # Start red color for output

        streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)
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
                    generation_config=GenerationConfig(
                        max_new_tokens=2,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                    ),
                    highlight_output=True,
                )
            )
        print("===============================")
        return generated_text


@YAMLConfigParser.register_metric
class Prompts(Interactive):
    prompts = [
        "What is gravity?",
        "What is a llama?",
        "Write a short story about a person who discovers a hidden room in their house. The story should include a plot twist and a clear resolution at the end.",
    ]

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
        for prompt in Prompts.prompts:
            print("===============================")
            generated_text.append(
                cls.generate_output(model, tokenizer, unformatted_prompt=prompt)
            )
        print("===============================")
        return generated_text
