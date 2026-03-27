from collections import Counter, defaultdict
from math import ceil
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
from simcse import SimCSE
import numpy as np
import json
import torch
import os

import Task
from InputExample import InputExample


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
        if len(self.labels) == 0:
            return 0.0
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


def weight(similarity: float, confidence: float, weighting_option: str = 'sim'):
    if weighting_option == 'sim':
        return similarity  # weighted only by similarity
    elif weighting_option == 'uniform' or weighting_option == 'concat':
        return 1  # uniform weighted
    elif weighting_option == 's+c':
        return similarity + confidence
    elif weighting_option == 'c':
        return confidence
    elif weighting_option == 'sc':
        return similarity * confidence
    else:
        raise Exception("Not Implemented")


class PrimingModelWrapper:
    def __init__(self, model: MaskedLMWrapper, task: Task, batch_size: int = 1):
        self.model = model
        self.task = task
        self.batch_size = batch_size

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

    def get_scores_batch(self, inputs: List[str], labels: List[str]):
        result = []
        total_batches = ceil(len(inputs) / self.batch_size) if inputs else 0
        for input_chunk in tqdm(chunks(inputs, self.batch_size), total=total_batches):
            logits = self.model.get_token_logits_batch(input_chunk).detach().cpu()
            result += [self.get_scores(example_logits, labels) for example_logits in logits]
        return result

    def get_scores(self, logits: torch.tensor, labels: List[str]):
        logits = torch.softmax(logits, dim=-1)
        scores = {}

        for label in labels:
            label_ids = self.model.tokenizer.encode(" " + label, add_special_tokens=False)
            # assert len(label_ids) == 1, f"Label {label} corresponds to multiple tokens: {label_ids}"
            scores[label] = logits[label_ids[0]].item()

        # normalize to 1
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

    def weighted_classify(self, examples: List[InputExample], all_neighbours: List[List[Tuple[InputExample, float]]],
                          normalize: bool = False):

        all_avg_results = []
        for example, neighbours in zip(tqdm(examples), all_neighbours):
            score_sum = sum(score for _, score in neighbours)

            all_examples, all_priming_examples, all_scores = [], [], []

            for idx, (neighbour, score) in enumerate(neighbours):
                neighbour_score = score / score_sum

                all_examples.append(example)
                all_priming_examples.append([neighbour])
                all_scores.append(neighbour_score)

            results = zip(self.classify(all_examples, all_priming_examples, normalize=normalize), all_scores)
            avg_results = defaultdict(list)

            for result, score in results:
                for k, v in result.items():
                    avg_results[k].append(v * score)
            avg_results = {k: sum(v) for k, v in avg_results.items()}
            all_avg_results.append(avg_results)
        return all_avg_results

    def concat_classify(self, examples: List[InputExample], all_neighbours: List[List[Tuple[InputExample, float]]],
                        normalize: bool = False):

        result = self.classify(examples, [[neighbor for neighbor, _ in neighbors] for neighbors in all_neighbours],
                               normalize=normalize)

        return result

    def unlabeled_priming(self, ds_test, ds_train,
                          task_name, model_name="albert-xlarge-v2", embedder_name="paraphrase-MiniLM-L6-v2",
                          normalize=True, top_k=3, num_iteration=3, priming_method="uniform", confidence_threshold=0):

        princeton = False  # if use the sentence transformers from princeton nlp
        if embedder_name.startswith("princeton-nlp"):
            embedder = SimCSE(embedder_name)
            embedder_name = embedder_name[embedder_name.index('/') + 1:]
            princeton = True
        else:
            embedder = SentenceTransformer(embedder_name)

        if top_k == 0:  # top_k = 0 means without priming
            print("Predicting for the test examples without priming:")
            return self.classify(ds_test, None, normalize=normalize)

        print(f"task_name={task_name}\n" +
              f"model_name={model_name} embedder_name={embedder_name}\n" +
              f"normalize={normalize} priming_method={priming_method}\n" +
              f"num_neighbors={top_k} num_iteration={num_iteration} confidence_threshold={confidence_threshold}\n")

        if princeton:
            train_embeddings = embedder.encode([x.text_a + " " + x.text_b for x in ds_train])
        else:
            train_embeddings = embedder.encode([x.text_a + " " + x.text_b for x in ds_train], convert_to_tensor=True,
                                               show_progress_bar=True)

        print("Predicting for the unlabeled examples:")
        file_name = 'data/' + task_name + '/' + model_name + '/' + embedder_name + '/' + \
                    str(num_iteration) + "iteration_" + str(len(ds_train)) + '.pt'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if os.path.exists(file_name):
            print("Loading the saved file...")
            unlabeled_examples_class_dist = torch.load(file_name)
            print("Successfully loaded.")
        else:
            unlabeled_examples_class_dist = self.classify(ds_train, None, normalize=normalize)
            # iteratively using unlabeled priming to improve the predictions for unlabeled examples
            for i in range(num_iteration):
                print(f"{i + 1}th iteration:\n")
                ds_train = [InputExample(example.text_a, example.text_b, max(unlabeled_examples_class_dist[idx],
                                                                             key=unlabeled_examples_class_dist[idx].get))
                            for idx, example in enumerate(tqdm(ds_train))]
                unlabeled_examples_class_dist = self.inference(ds_train, ds_train, unlabeled_examples_class_dist,
                                                               train_embeddings=train_embeddings,
                                                               embedder=embedder, princeton=princeton,
                                                               normalize=normalize, top_k=top_k,
                                                               priming_method=priming_method)
            torch.save(unlabeled_examples_class_dist, file_name)

        # optionally filter out unconfident examples
        if confidence_threshold > 0:
            ds_train = [InputExample(example.text_a, example.text_b, max(unlabeled_examples_class_dist[idx],
                                                                         key=unlabeled_examples_class_dist[idx].get))
                        for idx, example in enumerate(tqdm(ds_train))
                        if max(unlabeled_examples_class_dist[idx].values()) >= confidence_threshold]
            train_embeddings = None
        else:
            ds_train = [InputExample(example.text_a, example.text_b, max(unlabeled_examples_class_dist[idx],
                                                                         key=unlabeled_examples_class_dist[idx].get))
                        for idx, example in enumerate(tqdm(ds_train))]

        return self.inference(ds_test, ds_train, unlabeled_examples_class_dist,
                              train_embeddings=train_embeddings,
                              embedder=embedder, princeton=princeton,
                              normalize=normalize, top_k=top_k, priming_method=priming_method)

    def inference(self, ds_test, ds_train, unlabeled_examples_class_dist, train_embeddings,
                  embedder, princeton: bool,
                  normalize=True, top_k=3, priming_method="uniform"):

        if train_embeddings is None:
            if princeton:
                train_embeddings = embedder.encode([x.text_a + " " + x.text_b for x in ds_train])
            else:
                train_embeddings = embedder.encode([x.text_a + " " + x.text_b for x in ds_train],
                                                   convert_to_tensor=True,
                                                   show_progress_bar=True)

        print("Preparing the neighbors for the input examples:")
        top_k = min(top_k, len(ds_train))
        ds_test = tqdm(ds_test, desc="Preparing the neighbors")
        neighbours = defaultdict(list)
        for qidx, test_example in enumerate(ds_test):
            if princeton:
                query_embedding = embedder.encode(test_example.text_a + " " + test_example.text_b)
            else:
                query_embedding = embedder.encode(test_example.text_a + " " + test_example.text_b,
                                                  convert_to_tensor=True, show_progress_bar=False)

            cos_scores = util.pytorch_cos_sim(query_embedding, train_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            for similarity, idx in zip(top_results[0], top_results[1]):
                confidence = max(unlabeled_examples_class_dist[idx].values())
                score = weight(similarity, confidence, priming_method)
                neighbours[qidx].append((ds_train[idx], score))

        print("Predicting for the test examples with priming:")
        if priming_method != "concat":
            return self.weighted_classify(ds_test, neighbours.values(), normalize=normalize)
        else:
            return self.concat_classify(ds_test, neighbours.values(), normalize=normalize)
