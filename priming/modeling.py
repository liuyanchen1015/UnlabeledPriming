from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer, util
import random
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class TestResult:
    def __init__(self, scores: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None,
                 num_labels: Optional[int] = None):
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
    text_b: Optional[str] = ""
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


class YelpTask(Task):
    MAX_TOKENS_PER_EXAMPLE = 128

    LABEL_MAP = {
        0: "terrible",
        1: "bad",
        2: "ok",
        3: "good",
        4: "great"
    }

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        if subset == "train":
            dataset = datasets.load_dataset("yelp_review_full",split='train[:10000]')
        else:
            dataset = datasets.load_dataset("yelp_review_full", split='test[:1000]')
        examples = [self._convert_example(example) for example in dataset]
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
    MAX_TOKENS_PER_EXAMPLE = 128

    LABEL_MAP = {
        0: "bad",
        1: "good",
    }

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
            text_a = example['sentence'].replace("\n", " ").replace("<br />", " ")
        else:
            text_a = example['text'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:IMDBTask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)
        return InputExample(text_a=text_a, label=IMDBTask.LABEL_MAP[example['label']])


class SST2Task(Task):
    MAX_TOKENS_PER_EXAMPLE = 128

    LABEL_MAP = {
        0: "bad",
        1: "good",
    }

    Label_Map_Test = {
        -1: "bad",
        1: "good"
    }

    def load_dataset(self, subset) -> List[InputExample]:
        dataset = datasets.load_dataset("glue", "sst2")
        examples = [self._convert_example(example, subset) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [SST2Task.LABEL_MAP[idx] for idx in sorted(SST2Task.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}\nThe movie is{label_str}"

    def _convert_example(self, example: Dict[str, Any], subset) -> InputExample:
        text_a = example['sentence'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:SST2Task.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)
        if subset == "train":
            return InputExample(text_a=text_a, label=SST2Task.LABEL_MAP[example['label']])
        else:
            return InputExample(text_a=text_a, label=SST2Task.Label_Map_Test[example['label']])


class MNLITask(Task):
    MAX_TOKENS_PER_EXAMPLE = 128

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

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        if subset == "train":
            dataset = datasets.load_dataset("glue", "mnli", split='train[:10000]')
        else:
            dataset = datasets.load_dataset("glue", "mnli_matched", split='test[:1000]')

        examples = [self._convert_example(example, subset) for example in dataset]
        return examples

    def get_labels(self) -> List[str]:
        return [MNLITask.LABEL_MAP[idx] for idx in sorted(MNLITask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}?\n{label_str}, {example.text_b}"

    def _convert_example(self, example: Dict[str, Any], subset: str) -> InputExample:
        text_a = example['premise'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:MNLITask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)

        text_b = example['hypothesis'].replace("\n", " ").replace("<br />", " ")
        text_b = self.tokenizer.encode(text_b, add_special_tokens=False)
        text_b = text_b[:MNLITask.MAX_TOKENS_PER_EXAMPLE]
        text_b = self.tokenizer.decode(text_b)
        if subset == "train":
            return InputExample(text_a=text_a, text_b=text_b, label=MNLITask.LABEL_MAP[example['label']])
        else:
            return InputExample(text_a=text_a, text_b=text_b, label=MNLITask.Label_Map_Test[example['label']])


class RTETask(Task):
    MAX_TOKENS_PER_EXAMPLE = 128

    LABEL_MAP = {
        0: "Wrong",
        1: "Right"
    }

    Label_Map_Test = {
        -1: "Wrong",
        1: "Right"
    }

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        dataset = datasets.load_dataset("glue", "rte")
        examples = [self._convert_example(example, subset) for example in dataset[subset]]
        return examples

    def get_labels(self) -> List[str]:
        return [RTETask.LABEL_MAP[idx] for idx in sorted(RTETask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}?\n{label_str}, {example.text_b}"

    def _convert_example(self, example: Dict[str, Any], subset) -> InputExample:
        text_a = example['sentence1'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:RTETask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)

        text_b = example['sentence2'].replace("\n", " ").replace("<br />", " ")
        text_b = self.tokenizer.encode(text_b, add_special_tokens=False)
        text_b = text_b[:RTETask.MAX_TOKENS_PER_EXAMPLE]
        text_b = self.tokenizer.decode(text_b)
        if subset == "train":
            return InputExample(text_a=text_a, text_b=text_b, label=RTETask.LABEL_MAP[example['label']])
        else:
            return InputExample(text_a=text_a, text_b=text_b, label=RTETask.Label_Map_Test[example['label']])


class CSQATask(Task):
    MAX_TOKENS_PER_EXAMPLE = 128

    LABEL_MAP = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E"
    }

    def load_dataset(self, subset: str = "train") -> List[InputExample]:
        if subset == "train":
            dataset = datasets.load_dataset("commonsense_qa", split='train[:8741]')
        else:
            dataset = datasets.load_dataset("commonsense_qa", split='train[8741:]')

        examples = [self._convert_example(example) for example in dataset]
        return examples

    def get_labels(self) -> List[str]:
        return [CSQATask.LABEL_MAP[idx] for idx in sorted(CSQATask.LABEL_MAP.keys())]

    def format_example(self, example: InputExample, label_str: str) -> str:
        return f"{example.text_a}?\n{example.text_b}Answer is{label_str}"

    def _convert_example(self, example: Dict[str, Any]) -> InputExample:
        text_a = example['question'].replace("\n", " ").replace("<br />", " ")
        text_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        text_a = text_a[:CSQATask.MAX_TOKENS_PER_EXAMPLE]
        text_a = self.tokenizer.decode(text_a)

        text_b = ""
        for idx in range(len(CSQATask.LABEL_MAP.keys())):
            text_b += CSQATask.LABEL_MAP[idx] + ": " + example['choices']['text'][idx] + ". "
        text_b = self.tokenizer.encode(text_b, add_special_tokens=False)
        text_b = text_b[:CSQATask.MAX_TOKENS_PER_EXAMPLE]
        text_b = self.tokenizer.decode(text_b)

        return InputExample(text_a=text_a, text_b=text_b, label=example['answerKey'])



class PrimingModelWrapper:
    def __init__(self, model: MaskedLMWrapper, task: Task):
        self.model = model
        self.task = task

    def create_priming_prefix(self, example: InputExample, priming_examples: List[InputExample],
                              example_separator: str = "\n\n") -> str:
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

        #  normalize to 1
        score_sum = sum(scores.values())
        scores = {k: v / score_sum for k, v in scores.items()}

        return scores

    def classify(self, examples: List[InputExample], priming_examples: Optional[List[List[InputExample]]],
                 normalize: bool = False):

        if not priming_examples:
            priming_examples = [[] for _ in examples]

        inputs = [self.create_priming_prefix(example, priming_exs) for example, priming_exs in
                  zip(examples, priming_examples)]

        if normalize:
            baseline_inputs = [self.create_priming_prefix(InputExample("", "", ""), priming_exs) for priming_exs in
                               priming_examples]
            return self.get_normalized_scores(inputs, baseline_inputs, labels=self.task.get_labels())
        else:
            return self.get_scores_batch(inputs, labels=self.task.get_labels())

    """
    For each given neighbor example calculate the label distribution for the given input example,
    then weight the label distribution according to the similarities of the neighbors.
    """
    def weighted_classify(self, example: InputExample, neighbours: List[Tuple[InputExample, float]],
                          normalize: bool = False):
        # if the model could already make the confident prediction without any in-context examples,
        # then keep this prediction
        example_prediction = self.classify([example], [[]], normalize=normalize)[0]
        if not neighbours or max(example_prediction.values()) >= 0.8:
            return example_prediction

        score_sum = sum(score for _, score in neighbours)
        neighbour_class_dist = self.classify([ex for ex, _ in neighbours], None, normalize=normalize)

        all_examples, all_priming_examples, all_scores = [], [], []

        for idx, (neighbour, score) in enumerate(neighbours):
            neighbour_score = score / score_sum
            neighbour_class = max(neighbour_class_dist[idx], key=neighbour_class_dist[idx].get)
            priming_example = InputExample(neighbour.text_a, neighbour.text_b, neighbour_class)

            all_examples.append(example)
            all_priming_examples.append([priming_example])
            all_scores.append(neighbour_score)

        results = zip(self.classify(all_examples, all_priming_examples, normalize=normalize), all_scores)
        avg_results = defaultdict(list)

        for result, score in results:
            for k, v in result.items():
                avg_results[k].append(v * score.item())
        avg_results = {k: sum(v) for k, v in avg_results.items()}
        return avg_results

    """
    Priming all the neighbors at once as the context, and directly calculate the label distribution for input example.
    """
    def unweighted_classify(self, example: InputExample, neighbours: List[Tuple[InputExample, float]],
                            normalize: bool = False):  # unweighted classify
        # if the model could already make the confident prediction without any in-context examples,
        # then keep this prediction
        example_prediction = self.classify([example], [[]], normalize=normalize)[0]
        if not neighbours or max(example_prediction.values()) >= 0.8:
            return example_prediction

        # calculate the class distribution for each neighbor example (unlabeled priming)
        neighbour_class_dist = self.classify([ex for ex, _ in neighbours], None, normalize=normalize)

        all_priming_examples = []
        for idx, (neighbour, score) in enumerate(neighbours):
            neighbour_class = max(neighbour_class_dist[idx], key=neighbour_class_dist[idx].get)
            priming_example = InputExample(neighbour.text_a, neighbour.text_b, neighbour_class)
            all_priming_examples.append(priming_example)

        result = self.classify([example], [all_priming_examples], normalize=normalize)[0]

        return result

    def predict(self, ds_test, ds_train, embedder_name="paraphrase-MiniLM-L6-v2", normalize=True, priming=True,
                top_k=10, weighted=True, filter_unconfident=False, confidence_threshold=0, similarity_threshold=0):

        embedder = SentenceTransformer(embedder_name)

        if not priming:
            print("Predict for the test examples:")
            examples = tqdm(ds_test)
            if weighted:
                return [self.weighted_classify(example, [], normalize=normalize) for example in examples]
            else:
                return [self.unweighted_classify(example, [], normalize=normalize) for example in examples]

        print(f"normalize={normalize} num_neighbors={top_k}\n" +
              f"weighted={weighted} filter_unconfident={filter_unconfident}\n" +
              f"confidence_threshold={confidence_threshold} similarity_threshold={similarity_threshold}\n")

        print("Predict for the unlabeled examples:")
        # Predict for the unlabeled examples and optionally filter out unconfident ones.
        neighbour_class_dist = [self.classify([example], None, normalize=normalize)[0] for example in tqdm(ds_train)]

        if filter_unconfident:
            ds_train = [InputExample(example.text_a, example.text_b, max(neighbour_class_dist[idx],
                                                                         key=neighbour_class_dist[idx].get))
                        for idx, example in enumerate(ds_train)
                        if max(neighbour_class_dist[idx].values()) >= confidence_threshold]
        else:
            ds_train = [InputExample(example.text_a, example.text_b, max(neighbour_class_dist[idx],
                                                                         key=neighbour_class_dist[idx].get))
                        for idx, example in enumerate(ds_train)]

        # print(len(ds_train))

        top_k = min(top_k, len(ds_train))
        train_embeddings = embedder.encode([x.text_a + " " + x.text_b for x in ds_train], convert_to_tensor=True, show_progress_bar=True)

        print("Prepare the neighbors for the input examples:")
        ds_test = tqdm(ds_test, desc="Preparing the neighbors")
        neighbours = defaultdict(list)
        for qidx, test_example in enumerate(ds_test):
            query_embedding = embedder.encode(test_example.text_a + " " + test_example.text_b , convert_to_tensor=True, show_progress_bar=False)
            cos_scores = util.pytorch_cos_sim(query_embedding, train_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            for score, idx in zip(top_results[0], top_results[1]):
                if score < similarity_threshold:
                    break
                # weighted only by similarity:
                neighbours[qidx].append((ds_train[idx], score))

                # weighted by similarity x confidence:
                # neighbours[qidx].append((ds_train[idx], score * max(neighbour_class_dist[idx].values())))

        print("Predict for the test examples:")
        examples = tqdm(ds_test)
        if weighted:
            return [self.weighted_classify(example, neighbours.get(idx), normalize=normalize)
                    for idx, example in enumerate(examples)]
        else:
            return [self.unweighted_classify(example, neighbours.get(idx), normalize=normalize)
                    for idx, example in enumerate(examples)]


if __name__ == '__main__':
    model = MaskedLMWrapper("albert-xlarge-v2")
    task = SST2Task(tokenizer=model.tokenizer)
    priming_model_wrapper = PrimingModelWrapper(model, task)
    num_test_examples = 1000
    num_unlabeled_examples = 10000
    ds_train = task.load_dataset("train")
    ds_test = task.load_dataset("test")

    rng = random.Random(42)
    rng.shuffle(ds_train)
    rng.shuffle(ds_test)

    ds_train = ds_train[:num_unlabeled_examples]
    ds_test = ds_test[:num_test_examples]

    all_example_scores = priming_model_wrapper.predict(ds_test, ds_train, priming=False)

    test_result = TestResult(num_labels=len(task.get_labels()))
    all_example_scores = tqdm(all_example_scores, desc="Eval (Acc=??.?)")
    for idx, example_scores in enumerate(all_example_scores):
        label = priming_model_wrapper.task.get_labels().index(ds_test[idx].label)
        score = [example_scores[label] for label in priming_model_wrapper.task.get_labels()]

        test_result.add(np.array([score]), np.array([label]))
        all_example_scores.set_description(f"Eval (Acc={test_result.acc() * 100:4.1f})", refresh=True)

    print(f"Result: Acc={test_result.acc()} | LD={test_result.label_distribution()}")
