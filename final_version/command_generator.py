"""Generate reproducible experiment command grids."""

from typing import Iterable

DEFAULT_MODEL = "albert-xlarge-v2"
DEFAULT_EMBEDDER = "paraphrase-MiniLM-L6-v2"


def construct_command(model_name: str, embedder_name: str, task_name: str, top_k: int,
                      confidence_threshold: float, priming_method: str) -> str:
    command = ["python run_experiment.py", "-n"]

    if model_name != DEFAULT_MODEL:
        command.extend(["-m", model_name])
    if embedder_name != DEFAULT_EMBEDDER:
        command.extend(["-e", embedder_name])

    command.extend(["-t", task_name, "-k", str(top_k)])

    if top_k != 0:
        command.extend(["-c", str(confidence_threshold), "-p", priming_method])

    return " ".join(command) + "\n"


def write_commands(file_obj, task_names: Iterable[str]) -> None:
    for task_name in task_names:
        file_obj.write(construct_command(DEFAULT_MODEL, DEFAULT_EMBEDDER, task_name, 0, 0, "uniform"))
        for k in [3, 10, 50]:
            for c in [0, 0.8]:
                for priming_method in ["sim", "uniform"]:
                    file_obj.write(construct_command(DEFAULT_MODEL, DEFAULT_EMBEDDER, task_name, k, c, priming_method))
        for c in [0, 0.8]:
            file_obj.write(construct_command(DEFAULT_MODEL, DEFAULT_EMBEDDER, task_name, 3, c, "concat"))


if __name__ == '__main__':
    with open("experiments.txt", 'w', encoding='utf-8') as f:
        f.write("# main experiments\n")
        write_commands(f, ["agnews", "imdb", "sst2", "yahoo", "yelp", "boolq"])

        f.write("\n# experiments about priming method\n")
        for k in [3, 10, 50]:
            for priming_method in ["s+c", "sc"]:
                f.write(construct_command(DEFAULT_MODEL, DEFAULT_EMBEDDER, "agnews", k, 0, priming_method))

        f.write("\n# experiments about confidence threshold\n")
        for k in [3, 10]:
            for c in [0.75, 0.85]:
                f.write(construct_command(DEFAULT_MODEL, DEFAULT_EMBEDDER, "agnews", k, c, "sim"))

        f.write("\n# experiments about underlying LM and Sentence Transformer\n")
        f.write("# priming case: in setting k=3, c=0, similarity weighted\n")

        for task_name in ["agnews", "yahoo", "imdb"]:
            for model_name in ["albert-xxlarge-v2", "roberta-large", "bert-large-uncased"]:
                f.write(construct_command(model_name, DEFAULT_EMBEDDER, task_name, 3, 0, "sim"))

            f.write(construct_command("albert-xlarge-v2", "multi-qa-mpnet-base-dot-v1", task_name, 3, 0, "sim"))

            for embedder_name in [
                "multi-qa-mpnet-base-dot-v1",
                "multi-qa-MiniLM-L6-cos-v1",
                "all-mpnet-base-v2",
                "msmarco-bert-base-dot-v5",
                "princeton-nlp/unsup-simcse-roberta-large",
            ]:
                f.write(construct_command("albert-xxlarge-v2", embedder_name, task_name, 3, 0, "sim"))

        f.write("# without priming case (independent on the sentence transformer)\n")
        for task_name in ["agnews", "yahoo", "imdb"]:
            for model_name in ["albert-xxlarge-v2", "roberta-large", "bert-large-uncased"]:
                f.write(construct_command(model_name, DEFAULT_EMBEDDER, task_name, 0, 0, "uniform"))
