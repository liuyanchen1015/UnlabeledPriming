from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class TestResult:
    def __init__(self, scores: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None, num_labels: Optional[int] = None):
        self.scores = scores if scores is not None else np.zeros([0, num_labels])
        self.labels = labels if labels is not None else np.zeros([0])
        self.predictions = np.argmax(self.scores, axis=-1)

    def add(self, scores: np.ndarray, labels: np.ndarray):
        self.scores = np.concatenate([self.scores, scores])
        self.labels = np.concatenate([self.labels, labels])
        self.predictions = np.concatenate([self.predictions, np.argmax(scores, axis=-1)])

    def acc(self) -> float:
        return (self.labels == self.predictions).sum() / len(self.labels)

    def label_distribution(self) -> Dict[int, int]:
        return Counter(self.predictions)


class MaskedLMWrapper:
    """Wrapper for a masked language model to perform mask infilling."""

    def __init__(self, model_name: str, use_cuda: bool = True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)

    def get_token_logits_batch(self, input_texts: List[str]) -> torch.Tensor:
        """For a batch of texts with one mask token each, return the logits for the masked position."""
        batch = self.tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        mask_positions = (batch['input_ids'] == self.tokenizer.mask_token_id)
        assert torch.all(mask_positions.sum(axis=-1) == 1), \
            "Each input text must contain exactly one mask token"

        scores = self.model(**batch)['logits']
        return scores[mask_positions]


@dataclass
class InputExample:
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


class Task(ABC):
    """Represents a task by providing access to its train/test/dev sets and methods for formatting examples."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        pass

    @abstractmethod
    def format_example(self, example: InputExample, label_str: str) -> str:
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        pass


class AgNewsTask(Task):
    MAX_TOKENS_PER_EXAMPLE = 128

    LABEL_MAP = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science"
    }

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        dataset = datasets.load_dataset("ag_news")
        examples = [self._convert_example(example) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [AgNewsTask.LABEL_MAP[idx] for idx in sorted(AgNewsTask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}\nNews Category:{label_str}"

    def _convert_example(self, example: Dict[str, Any]) -> InputExample:
        text_a = example['text'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:AgNewsTask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)
        return InputExample(text_a=text_a, label=AgNewsTask.LABEL_MAP[example['label']])


class PrimingModelWrapper:
    def __init__(self, model: MaskedLMWrapper, task: Task):
        self.model = model
        self.task = task

    def create_priming_prefix(self, example: InputExample, priming_examples: List[InputExample], example_separator: str = "\n\n") -> str:
        result = ''
        for priming_example in priming_examples:
            result += self.format_example(priming_example, with_label=True) + example_separator

        result += self.format_example(example, with_label=False)
        return result

    def format_example(self, example: InputExample, with_label: bool = True):
        mask_str = " " + self.model.tokenizer.mask_token
        label_str = mask_str if not with_label else (" " + example.label)
        return self.task.format_example(example, label_str)

    def get_normalized_scores(self, inputs: List[str], baseline_inputs: List[str], labels: List[str],
                              all_baseline_scores=None):

        all_scores = self.get_scores_batch(inputs, labels)

        if all_baseline_scores is None:
            all_baseline_scores = self.get_scores_batch(baseline_inputs, labels)

        result = []
        for scores, baseline_scores in zip(all_scores, all_baseline_scores):
            normalized_scores = {k: v / baseline_scores[k] for k, v in scores.items()}
            score_sum = sum(normalized_scores.values())
            result.append({k: v / score_sum for k, v in normalized_scores.items()})
        return result

    def get_scores_batch(self, inputs: List[str], labels: List[str], max_batch_size: int = 1):
        result = []
        for input_chunk in chunks(inputs, max_batch_size):
            logits = self.model.get_token_logits_batch(input_chunk).detach().cpu()
            result += [self.get_scores(example_logits, labels) for example_logits in logits]
        return result

    def get_scores(self, logits: torch.tensor, labels: List[str]):
        logits = torch.softmax(logits, dim=-1)
        scores = {}

        for label in labels:
            label_ids = self.model.tokenizer.encode(" " + label, add_special_tokens=False)
            assert len(label_ids) == 1, f"Label {label} corresponds to multiple tokens: {label_ids}"
            scores[label] = logits[label_ids[0]].item()

        return scores

    def classify(self, examples: List[InputExample], priming_examples: Optional[List[List[InputExample]]], normalize: bool = False):

        if not priming_examples:
            priming_examples = [[] for _ in examples]

        inputs = [self.create_priming_prefix(example, priming_exs) for example, priming_exs in zip(examples, priming_examples)]

        if normalize:
            baseline_inputs = [self.create_priming_prefix(InputExample("", "", ""), priming_exs) for priming_exs in priming_examples]
            return self.get_normalized_scores(inputs, baseline_inputs, labels=self.task.get_labels())
        else:
            return self.get_scores_batch(inputs, labels=self.task.get_labels())
