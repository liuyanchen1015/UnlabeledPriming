"""Experiment entrypoint for Unlabeled Priming.

This script orchestrates:
1) task loading
2) unlabeled priming / baseline inference
3) result aggregation and persistence
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type

import numpy as np

import Task
from Modeling import MaskedLMWrapper, PrimingModelWrapper, TestResult


TASK_REGISTRY: Dict[str, Type[Task.Task]] = {
    "agnews": Task.AgNewsTask,
    "yelp": Task.YelpTask,
    "imdb": Task.IMDBTask,
    "sst2": Task.SST2Task,
    "boolq": Task.BoolQTask,
    "yahoo": Task.YahooTask,
}


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    embedder_name: str
    task_name: str
    num_test_examples: int
    num_unlabeled_examples: int
    normalize: bool
    top_k: int
    num_iteration: int
    confidence_threshold: float
    priming_method: str
    batch_size: int


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run unlabeled priming experiments.")
    parser.add_argument("-m", "--model_name", type=str, default="albert-xlarge-v2", help="Underlying masked language model")
    parser.add_argument(
        "-e",
        "--embedder_name",
        type=str,
        default="paraphrase-MiniLM-L6-v2",
        help="Sentence embedding model",
    )
    parser.add_argument(
        "-t",
        "--task_name",
        type=str,
        default="agnews",
        help="Task name",
        choices=sorted(TASK_REGISTRY.keys()),
    )

    parser.add_argument("-nt", "--num_test_examples", type=int, default=sys.maxsize, help="Max number of test examples")
    parser.add_argument("-nu", "--num_unlabeled_examples", type=int, default=sys.maxsize, help="Max number of unlabeled examples")

    parser.add_argument("-n", "--normalize", action="store_true", help="Normalize class probabilities by baseline prompts")
    parser.add_argument("-k", "--top_k", type=int, default=3, help="Number of neighbors for priming; 0 means no priming")
    parser.add_argument("-i", "--num_iteration", type=int, default=3, help="Number of self-training iterations on unlabeled data")
    parser.add_argument("-c", "--confidence_threshold", type=float, default=0, help="Filter pseudo labels below this confidence")
    parser.add_argument(
        "-p",
        "--priming_method",
        type=str,
        default="uniform",
        choices=["concat", "uniform", "sim", "s+c", "sc", "c"],
        help="Neighbor weighting strategy",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for masked-LM scoring")

    args = parser.parse_args()
    return ExperimentConfig(**vars(args))


def build_task(config: ExperimentConfig, model: MaskedLMWrapper) -> Task.Task:
    task_cls = TASK_REGISTRY[config.task_name]
    return task_cls(tokenizer=model.tokenizer)


def run(config: ExperimentConfig) -> TestResult:
    model = MaskedLMWrapper(config.model_name)
    task = build_task(config, model)
    priming_model_wrapper = PrimingModelWrapper(model, task, config.batch_size)

    ds_train = task.load_dataset("train")
    ds_test = task.load_dataset("test")

    rng = random.Random(42)
    rng.shuffle(ds_train)
    rng.shuffle(ds_test)

    ds_train = ds_train[: config.num_unlabeled_examples]
    ds_test = ds_test[: config.num_test_examples]

    all_example_scores = priming_model_wrapper.unlabeled_priming(
        ds_test,
        ds_train,
        task_name=config.task_name,
        model_name=config.model_name,
        embedder_name=config.embedder_name,
        normalize=config.normalize,
        top_k=config.top_k,
        num_iteration=config.num_iteration,
        priming_method=config.priming_method,
        confidence_threshold=config.confidence_threshold,
    )

    test_result = TestResult(num_labels=len(task.get_labels()))
    labels = task.get_labels()
    for idx, example_scores in enumerate(all_example_scores):
        gold_index = labels.index(ds_test[idx].label)
        score = [example_scores[label] for label in labels]
        test_result.add(np.array([score]), np.array([gold_index]))

    return test_result


def build_result_path(config: ExperimentConfig) -> Path:
    embedder_name = config.embedder_name
    if embedder_name.startswith("princeton-nlp"):
        embedder_name = embedder_name[embedder_name.index("/") + 1 :]

    path = Path("results") / config.task_name / config.model_name / embedder_name

    prefix = "norm_" if config.normalize else ""
    if config.top_k == 0:
        filename = f"{prefix}without_priming.txt"
    else:
        filename = f"{prefix}{config.priming_method}_k{config.top_k}_c{config.confidence_threshold}.txt"

    return path / filename


def write_result(config: ExperimentConfig, test_result: TestResult) -> None:
    output_path = build_result_path(config)
    os.makedirs(output_path.parent, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"task_name={config.task_name}\n")
        f.write(f"model_name={config.model_name} embedder_name={config.embedder_name}\n")
        f.write(f"normalize={config.normalize} ")
        if config.top_k == 0:
            f.write("priming=False\n")
        else:
            f.write(
                "priming_method="
                f"{config.priming_method}\nnum_neighbors={config.top_k} "
                f"num_iteration={config.num_iteration} confidence_threshold={config.confidence_threshold}\n"
            )
        f.write(f"Result: Acc={test_result.acc()} | LD={test_result.label_distribution()}")


def main() -> None:
    config = parse_args()
    test_result = run(config)
    write_result(config, test_result)
    print(f"Result: Acc={test_result.acc()} | LD={test_result.label_distribution()}")


if __name__ == "__main__":
    main()
