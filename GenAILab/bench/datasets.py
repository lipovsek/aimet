# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Datasets for GenAI testing"""

from abc import ABC, abstractmethod
import ast
import gc
import re

from datasets import (
    Dataset as HFDataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, pipeline

from GenAILab.bench.determinism import set_seed
from GenAILab.bench.utils.generation_utils import build_generation_config
from GenAILab.bench.utils.prompt_utils import (
    CALIBRATION_PROMPTS_FILE,
    load_text_prompts,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser


class Dataset(ABC):
    """Generic GenAI Dataset class"""

    @staticmethod
    @abstractmethod
    def load_dataset(split: str):
        """Load dataset from huggingface"""

    @classmethod
    @abstractmethod
    def load_encoded_dataset(
        cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str
    ):
        """Load encoded and chunked dataset"""


class TextDataset(Dataset):
    """Dataset that accepts a tokenizer (PreTrainedTokenizer)."""


class MultimodalDataset(Dataset):
    """Dataset that accepts a processor (ProcessorMixin) for text + image inputs."""


class InterleavedDatasetWrapper(torch.utils.data.Dataset):
    """Interleaves entries from multiple datasets in round-robin order.

    Items are drawn alternately: dataset_0[0], dataset_1[0], dataset_0[1],
    dataset_1[1], ... Once a dataset is exhausted it is skipped, and the
    remaining datasets keep being drawn until all are exhausted. The total
    length is therefore the sum of all sub-dataset lengths (callers that want a
    smaller calibration set bound iteration separately, e.g. via the recipe's
    ``num_iterations``).
    """

    def __init__(self, datasets: list):
        self.datasets = datasets
        # Precompute the flat draw order as (dataset_idx, item_idx) pairs:
        # round-robin across datasets, skipping any that are exhausted, until
        # every dataset has been fully consumed.
        lengths = [len(d) for d in datasets]
        counters = [0] * len(datasets)
        self._order: list[tuple[int, int]] = []
        while any(counters[i] < lengths[i] for i in range(len(datasets))):
            for i in range(len(datasets)):
                if counters[i] < lengths[i]:
                    self._order.append((i, counters[i]))
                    counters[i] += 1

    def __len__(self):
        return len(self._order)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self._order)
        if not 0 <= index < len(self._order):
            raise IndexError(index)
        ds_idx, item_idx = self._order[index]
        return self.datasets[ds_idx][item_idx]


@YAMLConfigParser.register_dataset
class Interleaved(MultimodalDataset):
    """Meta-dataset that interleaves entries from multiple sub-datasets.

    YAML usage::

        dataset:
          name: Interleaved
          source_datasets:
            - name: Wikitext
              split: train
            - name: AOKVQA
              split: train

    Parent kwargs like ``image_size`` are automatically forwarded to
    sub-datasets whose ``load_encoded_dataset`` signature accepts them.
    """

    @staticmethod
    def load_dataset(split: str):
        raise NotImplementedError("Interleaved does not support load_dataset directly.")

    @classmethod
    def load_encoded_dataset(cls, tokenizer, context_length, source_datasets, **kwargs):
        if not source_datasets or not isinstance(source_datasets, list):
            raise ValueError(
                "Interleaved dataset requires a 'source_datasets' list with at least one entry."
            )
        loaded = []
        for sub_config in source_datasets:
            sub_config = sub_config.copy()
            sub_name = sub_config.pop("name")
            sub_cls = YAMLConfigParser.get_dataset(sub_name)
            # Text datasets need just the tokenizer; multimodal datasets
            # need the full processor and get image_size forwarded.
            if issubclass(sub_cls, TextDataset):
                sub_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            else:
                sub_tokenizer = tokenizer
                if "image_size" in kwargs:
                    sub_config.setdefault("image_size", kwargs["image_size"])
            loaded.append(
                sub_cls.load_encoded_dataset(
                    sub_tokenizer, context_length, **sub_config
                )
            )
        return InterleavedDatasetWrapper(loaded)


class ChunkedDataset(torch.utils.data.Dataset):
    """Internal helper class to chunk input IDs to static graph context length"""

    def __init__(self, tokenized_data: dict[str, torch.Tensor], size_per_chunk: int):
        self.tokenized_data = tokenized_data
        self.size_per_chunk = size_per_chunk

    def __len__(self):
        return len(self.tokenized_data["input_ids"][0]) // self.size_per_chunk

    def __getitem__(self, index: int):
        # Raise IndexError for out-of-range indices so that bare iteration over
        # this map-style dataset (e.g. itertools.islice without a DataLoader)
        # terminates correctly. Without this, an out-of-range slice silently
        # returns an empty (1, 0) tensor and iteration never stops.
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start_index = index * self.size_per_chunk
        end_index = (index + 1) * self.size_per_chunk
        return {
            "input_ids": self.tokenized_data["input_ids"][:, start_index:end_index],
            "attention_mask": self.tokenized_data["attention_mask"][
                :, start_index:end_index
            ],
        }


@YAMLConfigParser.register_dataset
class Wikitext(TextDataset):
    """Wikitest dataset"""

    @staticmethod
    def load_dataset(split: str):
        return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    @classmethod
    def load_encoded_dataset(
        cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str
    ):
        dataset_split = cls.load_dataset(split)
        join_token = (
            "\n\n"
            if split == "test" or tokenizer.bos_token is None
            else tokenizer.bos_token
        )
        encoded_dataset_split = tokenizer(
            join_token.join(dataset_split["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        )

        return ChunkedDataset(encoded_dataset_split, context_length)


@YAMLConfigParser.register_dataset
class TinyMMLU(TextDataset):
    """TinyMMLU dataset"""

    @staticmethod
    def load_dataset(split: str):
        return load_dataset("tinyBenchmarks/tinyMMLU", split=split)

    @classmethod
    def load_encoded_dataset(
        cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str
    ):
        dataset_split = cls.load_dataset(split)

        def tokenize(samples):
            tokenized_question = tokenizer(
                samples["input_formatted"],
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = tokenizer(
                list(map(lambda answer: chr(ord("A") + answer), samples["answer"])),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
                "input_formatted",
            ],
        )


@YAMLConfigParser.register_dataset
class MMLU(TextDataset):
    """MMLU Dataset"""

    @classmethod
    def _format_question(cls, question: str, choices: list[str]):
        return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    @classmethod
    def _format_question_and_answer(
        cls, question: str, choices: list[str], answer: str
    ):
        return cls._format_question(question, choices) + f" {answer}"

    @classmethod
    def load_fewshot(cls, num_fewshot: int = 5, fewshot_split: str = "dev"):
        if num_fewshot == 0:
            return {}

        fewshot_split = load_dataset("cais/mmlu", name="all", split=fewshot_split)
        grouped_fewshot_questions = {}

        def group_fewshot_questions(sample):
            question = sample["question"]
            choices = sample["choices"]
            subject = sample["subject"]
            answer = chr(ord("A") + sample["answer"])

            if len(grouped_fewshot_questions.get(subject, [])) >= num_fewshot:
                return

            if subject not in grouped_fewshot_questions:
                grouped_fewshot_questions[subject] = []

            grouped_fewshot_questions[subject].append(
                cls._format_question_and_answer(question, choices, answer)
            )

        fewshot_split.map(group_fewshot_questions)

        for subject, questions in grouped_fewshot_questions.items():
            if len(questions) < num_fewshot:
                raise ValueError(
                    f"Not enough samples available in split {fewshot_split} to satisfy {num_fewshot} fewshot samples."
                )

        def combine_questions(subject, questions):
            formatted_subject = subject.replace("_", " ")
            formatted_string = f"The following are multiple choice questions (with answers) about {formatted_subject}.\n\n"
            for question in questions:
                formatted_string += question
                formatted_string += "\n\n"
            return formatted_string

        formatted_fewshot_questions = {
            subject: combine_questions(subject, questions)
            for subject, questions in grouped_fewshot_questions.items()
        }
        return formatted_fewshot_questions

    @staticmethod
    def load_dataset(split: str = "test"):
        if split != "test":
            raise ValueError("MMLU dataset only supports test split.")
        return load_dataset("cais/mmlu", name="all", split=split)

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str,
        num_fewshot: int = 5,
        fewshot_split: str = "dev",
    ):
        dataset_split = cls.load_dataset(split)
        fewshot_subject_headers = cls.load_fewshot(num_fewshot, fewshot_split)

        def tokenize(sample):
            question = sample["question"]
            choices = sample["choices"]
            subject = sample["subject"]

            formatted_question = list(
                map(
                    lambda question, choices: cls._format_question(question, choices),
                    question,
                    choices,
                )
            )
            fewshot_formatted_question = (
                list(
                    map(
                        lambda subject, question: str(
                            fewshot_subject_headers[subject] + question
                        ),
                        subject,
                        formatted_question,
                    )
                )
                if num_fewshot > 0
                else formatted_question
            )

            tokenized_question = tokenizer(
                fewshot_formatted_question,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = tokenizer(
                list(map(lambda answer: chr(ord("A") + answer), sample["answer"])),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
            ],
        )


@YAMLConfigParser.register_dataset
class MMMLU(TextDataset):
    """MMLU Dataset"""

    @classmethod
    def _format_question(cls, question: str, choices: tuple[str]):
        return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    @classmethod
    def _format_question_and_answer(
        cls, question: str, choices: list[str], answer: str
    ):
        return cls._format_question(question, choices) + f" {answer}"

    @classmethod
    def load_fewshot(cls, dataset_split, num_fewshot: int = 5):
        if num_fewshot == 0:
            return {}

        grouped_fewshot_questions: dict[str, list[str]] = {}

        def group_fewshot_questions(sample: dict[str, str]):
            question = sample["Question"]
            choices = (sample["A"], sample["B"], sample["C"], sample["D"])
            subject = sample["Subject"]
            answer = sample["Answer"]

            # We need one extra question to make sure that we can create an appropriately formatted string even if one
            # of the fewshot questions is encountered.
            if len(grouped_fewshot_questions.get(subject, [])) >= num_fewshot + 1:
                return

            if subject not in grouped_fewshot_questions:
                grouped_fewshot_questions[subject] = []

            grouped_fewshot_questions[subject].append(
                cls._format_question_and_answer(question, choices, answer)
            )

        dataset_split.map(group_fewshot_questions)

        for subject, questions in grouped_fewshot_questions.items():
            if len(questions) < num_fewshot:
                raise ValueError(
                    f"Not enough samples available in split to satisfy {num_fewshot} fewshot samples."
                )

        return grouped_fewshot_questions

    @staticmethod
    def load_dataset(split: str = "default"):
        return load_dataset("openai/MMMLU", name=split, split="test")

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str,
        num_fewshot: int = 5,
    ):
        dataset_split = cls.load_dataset(split)
        grouped_fewshot_questions = cls.load_fewshot(dataset_split, num_fewshot)

        def tokenize(sample: dict[str, list[str]]):
            question = sample["Question"]
            A = sample["A"]
            B = sample["B"]
            C = sample["C"]
            D = sample["D"]
            subject = sample["Subject"]

            formatted_question = list(
                map(
                    lambda question, A, B, C, D: cls._format_question(
                        question, (A, B, C, D)
                    ),
                    question,
                    A,
                    B,
                    C,
                    D,
                )
            )

            def assemble_fewshot_question(formatted_question: str, subject: str):
                subject_fewshot_questions = grouped_fewshot_questions[subject]

                formatted_string = ""
                num_fewshot_questions_added = 0
                for fewshot_question in subject_fewshot_questions:
                    if num_fewshot_questions_added >= num_fewshot:
                        break
                    if formatted_question in fewshot_question:
                        continue

                    formatted_string += fewshot_question
                    formatted_string += "\n\n"
                    num_fewshot_questions_added += 1

                formatted_string += formatted_question
                return formatted_string

            fewshot_formatted_question = list(
                map(assemble_fewshot_question, formatted_question, subject)
            )

            tokenized_question = tokenizer(
                fewshot_formatted_question,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = tokenizer(
                sample["Answer"],
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "Question",
                "A",
                "B",
                "C",
                "D",
                "Answer",
                "Subject",
                "Unnamed: 0",
            ],
        )


class LazyMMMUDataset(torch.utils.data.Dataset):
    """Lazy MMMU dataset, to avoid ballooning memory costs once samples are processed."""

    def __init__(self, raw_dataset, processor, context_length, image_size=None):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.image_size = tuple(image_size) if image_size is not None else None

    def __len__(self):
        return len(self.raw_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        question = sample["question"]
        options = (
            ast.literal_eval(sample["options"])
            if isinstance(sample["options"], str)
            else sample["options"]
        )

        # Collect all images by slot (1-indexed)
        image_slots = {i: sample.get(f"image_{i}") for i in range(1, 8)}
        valid_images = [img for img in image_slots.values() if img is not None]

        # Format answer choices
        choices_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )

        # Build content blocks, splitting on <image N> placeholders.
        # Each reference (even repeated ones like <image 1> appearing twice)
        # becomes an image content block, and the corresponding image is
        # appended to ordered_images so the processor sees a 1:1 mapping
        # between placeholders and pixel data.
        content = []
        ordered_images = []
        parts = re.split(r"(<image \d+>)", question)
        for part in parts:
            m = re.match(r"<image (\d+)>", part)
            if m:
                img_idx = int(m.group(1))
                img = image_slots.get(img_idx)
                if img is not None:
                    content.append({"type": "image"})
                    ordered_images.append(img)
                else:
                    # Reference to a non-existent image — keep as text
                    content.append({"type": "text", "text": part})
            elif part.strip():
                content.append({"type": "text", "text": part})

        # If no placeholders were found but images exist, prepend them
        if not ordered_images and valid_images:
            for img in valid_images:
                content.insert(0, {"type": "image"})
            ordered_images = list(valid_images)

        content.append({"type": "text", "text": f"\n{choices_text}"})

        # "Answer:" in an assistant turn so continue_final_message continues
        # the model's response, not the user's text (bump MMMU.SCORING_VERSION if changed).
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "Answer:"},
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            enable_thinking=False,
        )
        if self.image_size is not None:
            ordered_images = [image.resize(self.image_size) for image in ordered_images]

        # Use the full processor with images so that image placeholder tokens
        # are expanded to match the actual image dimensions.
        inputs = self.processor(
            text=[text],
            images=ordered_images if ordered_images else None,
            return_tensors="pt",
        )

        # Truncate text tokens to context_length
        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]

        # Store answer and number of choices for the metric
        inputs["label"] = sample["answer"]
        inputs["num_options"] = len(options)

        # Remove token type IDS if they exist - recomputed on the fly
        inputs.pop("mm_token_type_ids", None)

        return inputs


@YAMLConfigParser.register_dataset
class MMMU(MultimodalDataset):
    """MMMU Dataset."""

    @staticmethod
    def load_dataset(split: str = "validation"):
        # Necessary because all the subjects are separate datasets
        # Some questions are free-response, but we only want to evaluate on multiple-choice
        configs = get_dataset_config_names("MMMU/MMMU")
        all_ds = [load_dataset("MMMU/MMMU", config, split=split) for config in configs]
        combined = concatenate_datasets(all_ds)
        return combined.filter(lambda x: x["question_type"] == "multiple-choice")

    @classmethod
    def load_encoded_dataset(
        cls, processor, context_length, split="validation", image_size=None
    ):
        raw_dataset = cls.load_dataset(split)
        return LazyMMMUDataset(
            raw_dataset, processor, context_length, image_size=image_size
        )


@YAMLConfigParser.register_dataset
class C4(TextDataset):
    """C4 dataset"""

    @staticmethod
    def load_dataset(split: str = "en", num_samples: int = 2048):
        stream = load_dataset("allenai/c4", name=split, split="train", streaming=True)
        return HFDataset.from_list(list(stream.take(num_samples)))

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str = "en",
        num_samples: int = 2048,
    ):
        dataset_split = cls.load_dataset(split, num_samples)
        join_token = "\n\n" if tokenizer.bos_token is None else tokenizer.bos_token
        encoded_dataset_split = tokenizer(
            join_token.join(dataset_split["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        )

        return ChunkedDataset(encoded_dataset_split, context_length)


class LazyAOKVQADataset(torch.utils.data.Dataset):
    """Lazy A-OKVQA dataset for multimodal backbone calibration.

    Each sample is an image + question processed through the full VLM processor,
    producing fused input_ids, attention_mask, pixel_values, and image_grid_thw.

    For the train split, the correct answer choice and the first rationale are
    appended as an assistant turn, giving the quantizer longer, more
    representative sequences to calibrate on.
    """

    def __init__(
        self,
        raw_dataset,
        processor,
        context_length,
        image_size=None,
        include_answer=False,
    ):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.image_size = tuple(image_size) if image_size is not None else None
        self.include_answer = include_answer

    def __len__(self):
        return len(self.raw_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        image = sample["image"]
        question = sample["question"]
        choices = sample["choices"]

        choices_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(choices)
        )

        content = [
            {"type": "image"},
            {"type": "text", "text": f"{question}\n{choices_text}\nAnswer:"},
        ]

        messages = [{"role": "user", "content": content}]

        if self.include_answer:
            answer_idx = sample["correct_choice_idx"]
            answer_letter = chr(65 + answer_idx)
            answer_text = f"{answer_letter}. {choices[answer_idx]}"
            rationales = sample.get("rationales", [])
            if rationales:
                answer_text += f"\n\nReasoning: {rationales[0]}"
            messages.append({"role": "assistant", "content": answer_text})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not self.include_answer,
        )

        if self.image_size is not None:
            image = image.resize(self.image_size)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]
        inputs.pop("mm_token_type_ids", None)

        return inputs


@YAMLConfigParser.register_dataset
class AOKVQA(MultimodalDataset):
    """A-OKVQA dataset for multimodal backbone calibration (CC BY 4.0)."""

    @staticmethod
    def load_dataset(split: str = "train"):
        return load_dataset("HuggingFaceM4/A-OKVQA", split=split)

    @classmethod
    def load_encoded_dataset(
        cls,
        processor,
        context_length,
        split="train",
        image_size=None,
    ):
        raw_dataset = cls.load_dataset(split)
        return LazyAOKVQADataset(
            raw_dataset,
            processor,
            context_length,
            image_size=image_size,
            include_answer=(split == "train"),
        )


@YAMLConfigParser.register_dataset
class GeneratedDataset(TextDataset):
    """Calibrate on text the float model generates from seed prompts. Generation
    runs through the standard HuggingFace ``AutoModelForCausalLM`` / ``generate``
    API in bfloat16, independent of the quantsim being calibrated.

    YAML usage::

        dataset:
          name: GeneratedDataset
          num_inputs: 5
          max_new_tokens: 512
          seed: 42

    ``model_id`` selects the float model to generate from. It defaults to the
    run's top-level ``model:`` (injected by the recipe chain) so calibration
    matches the quantized model, and need not appear in the dataset config; set
    it explicitly only to generate from a different model.
    """

    @staticmethod
    def load_dataset(split: str):
        raise NotImplementedError(
            "GeneratedDataset does not load from disk; it generates text from "
            "the float model. Use load_encoded_dataset."
        )

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        model_id: str,
        num_inputs: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
    ):
        prompts = load_text_prompts(prompts_file=CALIBRATION_PROMPTS_FILE)
        target_tokens = num_inputs * context_length

        encoded = cls._generate_encoded(
            tokenizer=tokenizer,
            model_id=model_id,
            prompts=prompts,
            target_tokens=target_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

        return cls._build_chunked_dataset(encoded, context_length)

    @staticmethod
    def _build_chunked_dataset(encoded, context_length):
        num_tokens = encoded["input_ids"].shape[-1]
        chunk_size = min(context_length, num_tokens)
        return ChunkedDataset(encoded, chunk_size)

    @classmethod
    def _generate_encoded(
        cls,
        *,
        tokenizer,
        model_id,
        prompts,
        target_tokens,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        seed,
    ) -> dict[str, torch.Tensor]:
        """Load the float model and generate until ``target_tokens`` is reached.

        Seed prompts are cycled in order; generation continues prompt-by-prompt
        until the concatenated, re-tokenized text has at least ``target_tokens``
        tokens, then it is truncated to exactly ``target_tokens``.
        """
        set_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        gen_config = build_generation_config(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )
        # The pipeline applies the chat template and generation prompt, runs
        # generate() under no-grad, and returns the prompt plus the generated
        # text (return_full_text=True), matching the original prompt+completion
        # calibration samples. Device placement comes from the model's
        # device_map, so no explicit device is passed.
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        join_token = "\n\n" if tokenizer.bos_token is None else tokenizer.bos_token

        generated_texts = []
        total_tokens = 0
        prompt_idx = 0
        # Cycle through the prompt list until we have enough tokens. A hard cap
        # (2x the prompts needed in the ideal case) guards against pathological
        # early-EOS loops that never reach target.
        max_generations = 2 * (target_tokens // max(max_new_tokens, 1) + len(prompts))
        progress = tqdm(
            total=target_tokens,
            unit="tok",
            desc=f"Generating calibration text ({model_id})",
        )
        try:
            while (
                total_tokens < target_tokens and len(generated_texts) < max_generations
            ):
                prompt = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ]
                # With chat input, return_full_text=True yields the full
                # conversation as a list of message dicts; render it back to a
                # single prompt+completion string via the chat template.
                full_messages = generator(
                    messages,
                    generation_config=gen_config,
                    return_full_text=True,
                )[0]["generated_text"]
                text = tokenizer.apply_chat_template(full_messages, tokenize=False)
                generated_texts.append(text)
                # Approximate running total; the authoritative count comes
                # from the final joined re-tokenization below.
                total_tokens = len(
                    tokenizer(
                        join_token.join(generated_texts),
                        add_special_tokens=True,
                    )["input_ids"]
                )
                # Clamp the bar to its total so a final overshoot doesn't
                # render as >100%.
                progress.n = min(total_tokens, target_tokens)
                progress.refresh()
        finally:
            progress.close()
            # Free the float weights before sim calibration runs.
            del generator
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        encoded = tokenizer(
            join_token.join(generated_texts),
            return_tensors="pt",
            add_special_tokens=True,
        )
        # Truncate to exactly target_tokens so the dataset is a whole number of
        # context-length chunks (when target_tokens is a multiple of CL).
        if encoded["input_ids"].shape[-1] > target_tokens:
            encoded = {
                "input_ids": encoded["input_ids"][:, :target_tokens],
                "attention_mask": encoded["attention_mask"][:, :target_tokens],
            }
        return encoded
