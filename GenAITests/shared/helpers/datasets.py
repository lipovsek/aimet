# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Datasets for GenAI testing"""

from abc import ABC, abstractmethod
import ast
import re

from datasets import (
    Dataset as HFDataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
import torch
from transformers import PreTrainedTokenizer

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser


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


class ChunkedDataset(torch.utils.data.Dataset):
    """Internal helper class to chunk input IDs to static graph context length"""

    def __init__(self, tokenized_data: dict[str, torch.Tensor], size_per_chunk: int):
        self.tokenized_data = tokenized_data
        self.size_per_chunk = size_per_chunk

    def __len__(self):
        return len(self.tokenized_data["input_ids"][0]) // self.size_per_chunk

    def __getitem__(self, index: int):
        start_index = index * self.size_per_chunk
        end_index = (index + 1) * self.size_per_chunk
        return {
            "input_ids": self.tokenized_data["input_ids"][:, start_index:end_index],
            "attention_mask": self.tokenized_data["attention_mask"][
                :, start_index:end_index
            ],
        }


@YAMLConfigParser.register_dataset
class Wikitext(Dataset):
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
class TinyMMLU(Dataset):
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
class MMLU(Dataset):
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
class MMMLU(Dataset):
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

    def __init__(self, raw_dataset, processor, context_length):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        question = sample["question"]
        options = (
            ast.literal_eval(sample["options"])
            if isinstance(sample["options"], str)
            else sample["options"]
        )

        # Collect all images
        images = [sample.get(f"image_{i}") for i in range(1, 8)]

        # Format answer choices
        choices_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )

        # Build content blocks, splitting on <image N> placeholders
        content = []
        parts = re.split(r"(<image \d+>)", question)
        for part in parts:
            if re.match(r"<image \d+>", part):
                content.append({"type": "image"})
            elif part.strip():
                content.append({"type": "text", "text": part})

        # If no placeholders were found but images exist, prepend them
        if not any(c.get("type") == "image" for c in content) and images:
            for _ in images:
                content.insert(0, {"type": "image"})

        content.append({"type": "text", "text": f"\n{choices_text}\nAnswer:"})

        # Apply chat template to get the text with image markers
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Use the full processor with images so that image placeholder tokens
        # are expanded to match the actual image dimensions.
        inputs = self.processor(
            text=[text],
            images=[image for image in images if image is not None],
            return_tensors="pt",
        )

        # Truncate text tokens to context_length
        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]

        # Tokenize the answer label
        tokenized_answer = self.tokenizer(
            sample["answer"],
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        inputs["label"] = tokenized_answer["input_ids"]

        return inputs


@YAMLConfigParser.register_dataset
class MMMU(Dataset):
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
    def load_encoded_dataset(cls, processor, context_length, split="validation"):
        raw_dataset = cls.load_dataset(split)
        return LazyMMMUDataset(raw_dataset, processor, context_length)


@YAMLConfigParser.register_dataset
class C4(Dataset):
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
