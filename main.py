"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from collections import defaultdict
from typing import List, Tuple, Dict
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from modeling import (
    InputExample,
    MaskedLMWrapper,
    AgNewsTask,
    TestResult,
    PrimingModelWrapper
)


def classify(priming_model_wrapper: PrimingModelWrapper, example: InputExample, neighbours: List[Tuple[InputExample, float]],
             normalize: bool = False):
    example_prediction = priming_model_wrapper.classify([example], [[]], normalize=normalize)[0]

    if not neighbours or max(example_prediction.values()) >= 0.8:
        return example_prediction

    score_sum = sum(score for _, score in neighbours)
    neighbour_class_dist = priming_model_wrapper.classify([ex for ex, _ in neighbours], None, normalize=normalize)

    all_examples, all_priming_examples, all_scores = [], [], []

    for idx, (neighbour, score) in enumerate(neighbours):
        neighbour_score = score / score_sum
        neighbour_class = max(neighbour_class_dist[idx], key=neighbour_class_dist[idx].get)
        priming_example = InputExample(neighbour.text_a, neighbour.text_b, neighbour_class)

        all_examples.append(example)
        all_priming_examples.append([priming_example])
        all_scores.append(neighbour_score)

    results = zip(priming_model_wrapper.classify(all_examples, all_priming_examples, normalize=normalize), all_scores)
    avg_results = defaultdict(list)

    for result, score in results:
        for k, v in result.items():
            avg_results[k].append(v * score.item())
    avg_results = {k: sum(v) for k, v in avg_results.items()}
    return avg_results


def test(priming_model_wrapper: PrimingModelWrapper, examples: List[InputExample], neighbours: Dict[int, List[Tuple[InputExample, float]]],
         normalize: bool = False) -> TestResult:
    scores, labels = [], []
    test_result = TestResult(num_labels=len(task.get_labels()))

    examples = tqdm(examples, desc="Eval (Acc=??.?)")

    for idx, example in enumerate(examples):
        example_scores = classify(priming_model_wrapper, example, neighbours.get(idx), normalize=normalize)
        labels.append(priming_model_wrapper.task.get_labels().index(example.label))
        scores.append([example_scores[label] for label in priming_model_wrapper.task.get_labels()])

        test_result.add(np.array([scores[-1]]), np.array([labels[-1]]))
        examples.set_description(f"Eval (Acc={test_result.acc() * 100:4.1f})", refresh=True)

    return TestResult(scores=np.array(scores), labels=np.array(labels))


if __name__ == '__main__':

    model = MaskedLMWrapper("albert-xlarge-v2")
    task = AgNewsTask(tokenizer=model.tokenizer)
    priming_model_wrapper = PrimingModelWrapper(model, task)
    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    num_test_examples = 1000
    num_unlabeled_examples = 10000

    ds_train = task.load_dataset("train")
    ds_test = task.load_dataset("test")

    rng = random.Random(42)
    rng.shuffle(ds_train)
    rng.shuffle(ds_test)

    ds_train = ds_train[:num_unlabeled_examples]
    ds_test = ds_test[:num_test_examples]

    train_embeddings = embedder.encode([x.text_a for x in ds_train], convert_to_tensor=True, show_progress_bar=True)
    top_k = min(3, len(ds_train))
    neighbours = defaultdict(list)

    for qidx, test_example in enumerate(ds_test):
        query_embedding = embedder.encode(test_example.text_a, convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.pytorch_cos_sim(query_embedding, train_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            neighbours[qidx].append((ds_train[idx], score))

    embedder = None

    for unlabeled_priming in [False, True]:
        test_result = test(priming_model_wrapper, ds_test, neighbours if unlabeled_priming else {}, normalize=True)
        print(f"Result (unlabeled_priming={unlabeled_priming}): Acc={test_result.acc()} | LD={test_result.label_distribution()}")
