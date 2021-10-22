from Modeling import PrimingModelWrapper, MaskedLMWrapper, TestResult
import argparse
import numpy as np
import random
import os
import sys
import Task


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model_name", type=str, default="albert-xlarge-v2", help="the underlying model")
    parser.add_argument("-e", "--embedder_name", type=str, default="paraphrase-MiniLM-L6-v2",
                        help="the underlying sentence embedding model")
    parser.add_argument("-t", "--task_name", type=str, default="agnews", help="the task",
                        choices=["agnews", "yelp", "imdb", "sst2", "mnli", "rte", "boolq", "yahoo"])

    parser.add_argument("-nt", "--num_test_examples", type=int, default=sys.maxsize, help="number of test examples")
    parser.add_argument("-nu", "--num_unlabeled_examples", type=int, default=10000, help="number of unlabeled examples")

    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the label distribution or not")
    parser.add_argument("-k", "--top_k", type=int, default=3,
                        help="the number of neighbors used for priming, 0 means without priming")
    parser.add_argument("-c", "--confidence_threshold", type=float, default=0, help="the confidence threshold")
    parser.add_argument("-p", "--priming_method", type=str, default="uniform", help="the method used for priming",
                        choices=["concat", "uniform", "sim", "s+c", "sc"])

    args = parser.parse_args()
    model_name = args.model_name
    model = MaskedLMWrapper(model_name)
    embedder_name = args.embedder_name
    priming_method = args.priming_method
    top_k = args.top_k

    if priming_method == 'concat':
        weighted = False  # concat all the neighbor examples and prime at once
    else:
        weighted = True  # prime one neighbor each time, and then weighted average the label distributions

    task_name = args.task_name
    if task_name == "agnews":
        task = Task.AgNewsTask(tokenizer=model.tokenizer, weighted=weighted)
    elif task_name == "yelp":
        task = Task.YelpTask(tokenizer=model.tokenizer, weighted=weighted)
    elif task_name == "imdb":
        task = Task.IMDBTask(tokenizer=model.tokenizer, weighted=weighted)
    elif task_name == "sst2":
        task = Task.SST2Task(tokenizer=model.tokenizer, weighted=weighted)
    elif task_name == "mnli":
        task = Task.MNLITask(tokenizer=model.tokenizer, weighted=weighted)
    elif task_name == "rte":
        task = Task.RTETask(tokenizer=model.tokenizer, weighted=weighted)
    elif task_name == "boolq":
        task = Task.BoolQTask(tokenizer=model.tokenizer, weighted=weighted)
    elif task_name == "yahoo":
        task = Task.YahooTask(tokenizer=model.tokenizer, weighted=weighted)
    priming_model_wrapper = PrimingModelWrapper(model, task)

    num_test_examples = args.num_test_examples
    num_unlabeled_examples = args.num_unlabeled_examples

    normalize = False
    if args.normalize:
        normalize = True

    confidence_threshold = args.confidence_threshold

    ds_train = task.load_dataset("train")
    ds_test = task.load_dataset("test")
    rng = random.Random(42)
    rng.shuffle(ds_train)
    rng.shuffle(ds_test)

    ds_train = ds_train[:num_unlabeled_examples]
    ds_test = ds_test[:num_test_examples]

    all_example_scores = priming_model_wrapper.inference(ds_test, ds_train,
                                                         task_name=task_name,
                                                         model_name=model_name, embedder_name=embedder_name,
                                                         normalize=normalize, top_k=top_k,
                                                         priming_method=priming_method,
                                                         confidence_threshold=confidence_threshold)

    test_result = TestResult(num_labels=len(task.get_labels()))
    for idx, example_scores in enumerate(all_example_scores):
        label = priming_model_wrapper.task.get_labels().index(ds_test[idx].label)
        score = [example_scores[label] for label in priming_model_wrapper.task.get_labels()]

        test_result.add(np.array([score]), np.array([label]))

    if embedder_name.startswith("princeton-nlp"):
        embedder_name = embedder_name[embedder_name.index('/') + 1:]
    file_name = 'results/' + task_name + '/' + model_name + '/' + embedder_name + '/'
    if normalize:
        file_name += 'norm_'

    if top_k == 0:  # without priming
        file_name += 'without_priming'
    else:  # priming
        file_name += priming_method + '_'
        file_name += 'k' + str(top_k) + '_'
        file_name += 'c' + str(confidence_threshold)
    file_name += '.txt'

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(f"task_name={task_name}\n" +
                f"model_name={model_name} embedder_name={embedder_name}\n" +
                f"normalize={normalize} ")
        if top_k == 0:  # without priming
            f.write(f"priming=False\n")
        else:  # priming
            f.write(f"priming_method={priming_method}\n" +
                    f"num_neighbors={top_k} confidence_threshold={confidence_threshold}\n")
        f.write(f"Result: Acc={test_result.acc()} | LD={test_result.label_distribution()}")

    print(f"Result: Acc={test_result.acc()} | LD={test_result.label_distribution()}")


if __name__ == '__main__':
    main()
