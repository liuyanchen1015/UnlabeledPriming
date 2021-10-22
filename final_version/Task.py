import datasets
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer
from InputExample import InputExample


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
    LABEL_MAP = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science"
    }

    MAX_TOKENS_PER_EXAMPLE = 120

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


class YahooTask(Task):
    LABEL_MAP = {
        0: "Society",
        1: "Science",
        2: "Health",
        3: "Education",
        4: "Computers",
        5: "Sports",
        6: "Business",
        7: "Entertainment",
        8: "Family",
        9: "Politics"
    }

    MAX_TOKENS_PER_EXAMPLE = 120

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        dataset = datasets.load_dataset("yahoo_answers_topics")
        examples = [self._convert_example(example) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [YahooTask.LABEL_MAP[idx] for idx in sorted(YahooTask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}\nNews Category:{label_str}"

    def _convert_example(self, example: Dict[str, Any]) -> InputExample:
        text = example['question_title'] + ' ' + example['question_content'] + ' ' + example['best_answer']
        text_a = text.replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:YahooTask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)
        return InputExample(text_a=text_a, label=YahooTask.LABEL_MAP[example['topic']])


class YelpTask(Task):
    LABEL_MAP = {
        0: "terrible",
        1: "bad",
        2: "ok",
        3: "good",
        4: "great"
    }

    MAX_TOKENS_PER_EXAMPLE = 120

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        dataset = datasets.load_dataset("yelp_review_full")
        examples = [self._convert_example(example) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [YelpTask.LABEL_MAP[idx] for idx in sorted(YelpTask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}\nIn summary, the restaurant is{label_str}"

    def _convert_example(self, example: Dict[str, Any]) -> InputExample:
        text_a = example['text'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:YelpTask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)
        return InputExample(text_a=text_a, label=YelpTask.LABEL_MAP[example['label']])


class IMDBTask(Task):
    LABEL_MAP = {
        0: "bad",
        1: "good",
    }

    MAX_TOKENS_PER_EXAMPLE = 120

    def load_dataset(self, subset) -> List[InputExample]:
        if subset == "train":
            dataset = datasets.load_dataset("glue", "sst2", split="train")
            examples = [self._convert_example(example, subset) for example in dataset]
        elif subset == "test":
            dataset = datasets.load_dataset("imdb", split="test")
            examples = [self._convert_example(example, subset) for example in dataset]
        return examples

    def get_labels(self) -> List[str]:
        return [IMDBTask.LABEL_MAP[idx] for idx in sorted(IMDBTask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}\nThe movie is{label_str}"

    def _convert_example(self, example: Dict[str, Any], subset) -> InputExample:
        if subset == 'train':
            text_a = example['sentence']
        else:
            text_a = example['text'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:IMDBTask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)
        return InputExample(text_a=text_a, label=IMDBTask.LABEL_MAP[example['label']])


class SST2Task(Task):
    LABEL_MAP = {
        0: "bad",
        1: "good",
    }

    Label_Map_Test = {
        -1: "bad",
        1: "good"
    }

    MAX_TOKENS_PER_EXAMPLE = 120

    def load_dataset(self, subset) -> List[InputExample]:
        dataset = datasets.load_dataset("glue", "sst2")
        examples = [self._convert_example(example, subset) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [SST2Task.LABEL_MAP[idx] for idx in sorted(SST2Task.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}\nThe movie is{label_str}"

    def _convert_example(self, example: Dict[str, Any], subset) -> InputExample:
        text_a = example['sentence']
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:SST2Task.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)
        if subset == "train":
            return InputExample(text_a=text_a, label=SST2Task.LABEL_MAP[example['label']])
        else:
            return InputExample(text_a=text_a, label=SST2Task.Label_Map_Test[example['label']])


class MNLITask(Task):
    LABEL_MAP = {
        0: "Wrong",
        1: "Right",
        2: "Maybe"
    }

    Label_Map_Test = {
        0: "Maybe",
        1: "Right",
        -1: "Wrong"
    }

    MAX_TOKENS_PER_EXAMPLE = 120

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        if subset == "train":
            dataset = datasets.load_dataset("glue", "mnli", split='train')
        else:
            dataset = datasets.load_dataset("glue", "mnli_matched", split='test')

        examples = [self._convert_example(example, subset) for example in dataset]
        return examples

    def get_labels(self) -> List[str]:
        return [MNLITask.LABEL_MAP[idx] for idx in sorted(MNLITask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}?\n{label_str}, {example.text_b}"

    def _convert_example(self, example: Dict[str, Any], subset: str) -> InputExample:
        text_a = example['premise']
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:int(MNLITask.MAX_TOKENS_PER_EXAMPLE / 2)]  # For each MNLI example, there are 2 sentence
        text_a = self.tokenizer.decode(text_a)

        text_b = example['hypothesis']
        text_b = self.tokenizer.encode(text_b, add_special_tokens=False)
        text_b = text_b[:int(MNLITask.MAX_TOKENS_PER_EXAMPLE / 2)]
        text_b = self.tokenizer.decode(text_b)
        if subset == "train":
            return InputExample(text_a=text_a, text_b=text_b, label=MNLITask.LABEL_MAP[example['label']])
        else:
            return InputExample(text_a=text_a, text_b=text_b, label=MNLITask.Label_Map_Test[example['label']])


class RTETask(Task):
    LABEL_MAP = {
        0: "Wrong",
        1: "Right"
    }

    Label_Map_Test = {
        -1: "Wrong",
        1: "Right"
    }

    MAX_TOKENS_PER_EXAMPLE = 120

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        dataset = datasets.load_dataset("glue", "rte")
        examples = [self._convert_example(example, subset) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [RTETask.LABEL_MAP[idx] for idx in sorted(RTETask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}?\n{label_str}, {example.text_b}"

    def _convert_example(self, example: Dict[str, Any], subset) -> InputExample:
        text_a = example['sentence1']
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:int(RTETask.MAX_TOKENS_PER_EXAMPLE / 2)]
        text_a = self.tokenizer.decode(text_a)

        text_b = example['sentence2']
        text_b = self.tokenizer.encode(text_b, add_special_tokens=False)
        text_b = text_b[:int(RTETask.MAX_TOKENS_PER_EXAMPLE / 2)]
        text_b = self.tokenizer.decode(text_b)
        if subset == "train":
            return InputExample(text_a=text_a, text_b=text_b, label=RTETask.LABEL_MAP[example['label']])
        else:
            return InputExample(text_a=text_a, text_b=text_b, label=RTETask.Label_Map_Test[example['label']])


class BoolQTask(Task):
    LABEL_MAP = {
        0: "false",
        1: "true",
    }

    MAX_TOKENS_PER_EXAMPLE = 120

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        dataset = datasets.load_dataset("boolq")
        if subset == 'test':
            subset = 'validation'
        examples = [self._convert_example(example) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [BoolQTask.LABEL_MAP[idx] for idx in sorted(BoolQTask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}. Based on the previous passage, {example.text_b}? {label_str}."

    def _convert_example(self, example: Dict[str, Any]) -> InputExample:
        text_a = example['passage'].replace("\n", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:int(BoolQTask.MAX_TOKENS_PER_EXAMPLE * 3 / 4)]
        text_a = self.tokenizer.decode(text_a)

        text_b = example['question'].replace("\n", " ")
        text_b = self.tokenizer.encode(text_b, add_special_tokens=False)
        text_b = text_b[:int(BoolQTask.MAX_TOKENS_PER_EXAMPLE * 1 / 4)]
        text_b = self.tokenizer.decode(text_b)

        return InputExample(text_a=text_a, text_b=text_b, label=BoolQTask.LABEL_MAP[example['answer']])
