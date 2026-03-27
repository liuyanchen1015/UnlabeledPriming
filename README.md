# Semantic-Oriented Unlabeled Priming for Large-Scale Language Models

Official (restructured) codebase for reproducing and extending the method in:

> **Semantic-Oriented Unlabeled Priming for Large-Scale Language Models**  
> arXiv:2202.06133  
> https://arxiv.org/abs/2202.06133

---

## 1. Overview

This repository implements an **unlabeled priming** framework for prompt-based text classification with masked language models (MLMs).

Core idea:
- Use a sentence encoder to retrieve semantically similar unlabeled examples.
- Iteratively pseudo-label unlabeled data.
- Prime each inference example with retrieved neighbors.
- Aggregate label scores with configurable weighting strategies.

Compared with plain zero-shot prompting (`k=0`), this method can improve predictive stability and accuracy without supervised fine-tuning.

---

## 2. Repository Structure

```text
.
├── final_version/
│   ├── run_experiment.py      # Main experiment entry point
│   ├── Modeling.py            # MLM wrapper, priming/inference pipeline
│   ├── Task.py                # Dataset/task definitions and prompt templates
│   ├── InputExample.py        # Shared data structure
│   ├── command_generator.py   # Generate command grids
│   └── requirements.txt
├── original_version/          # Legacy implementation snapshot
├── scripts/
│   ├── run_experiments.sh
│   └── run_experiments_batch.sh
└── README.md
```

---

## 3. Environment Setup

### 3.1 Python dependencies

```bash
cd final_version
pip install -r requirements.txt
```

### 3.2 Typical runtime requirements

- Python 3.8+
- CUDA GPU recommended (large MLMs and embedding models)
- Internet access for first-time model/dataset download (Hugging Face)

---

## 4. Quick Start

Run a baseline (no priming):

```bash
cd final_version
python run_experiment.py -n -t agnews -k 0
```

Run unlabeled priming with similarity weighting:

```bash
python run_experiment.py -n -t agnews -k 3 -i 3 -c 0 -p sim
```

Outputs are saved under:

```text
final_version/results/<task>/<mlm>/<embedder>/...
```

---

## 5. Main CLI Arguments

| Argument | Meaning |
|---|---|
| `-m, --model_name` | Masked language model (e.g., `albert-xlarge-v2`) |
| `-e, --embedder_name` | Sentence embedding model for retrieval |
| `-t, --task_name` | Task (`agnews`, `yelp`, `imdb`, `sst2`, `boolq`, `yahoo`) |
| `-k, --top_k` | Number of neighbors for priming (`0` = no priming) |
| `-i, --num_iteration` | Number of unlabeled pseudo-labeling iterations |
| `-c, --confidence_threshold` | Minimum pseudo-label confidence |
| `-p, --priming_method` | Weighting (`uniform`, `sim`, `concat`, `s+c`, `sc`, `c`) |
| `-n, --normalize` | Enable score normalization against baseline prompts |

---

## 6. Method Notes

### 6.1 Priming modes

- **`k=0`**: vanilla prompt-based zero-shot prediction.
- **`k>0`**: retrieve `k` nearest unlabeled examples and prime the prompt.

### 6.2 Weighting strategies

- `uniform`: equal weight for each neighbor.
- `sim`: weight by embedding similarity.
- `concat`: concatenate neighbors into one priming context.
- `c`: weight by pseudo-label confidence.
- `s+c` / `sc`: combined similarity-confidence heuristics.

---

## 7. Reproducibility Recommendations

- Keep random seed fixed (`42` in `run_experiment.py`).
- Report full config including MLM, embedder, `k`, `i`, `c`, and weighting.
- Record hardware and software versions.
- Save raw result files from `results/` for auditability.

---

## 8. Citation

If this repository or the method is useful in your work, please cite the paper:

```bibtex
@inproceedings{liu-etal-2023-semantic,
    title = "Semantic-Oriented Unlabeled Priming for Large-Scale Language Models",
    author = "Liu, Yanchen  and
      Schick, Timo  and
      Schtze, Hinrich",
    editor = "Sadat Moosavi, Nafise  and
      Gurevych, Iryna  and
      Hou, Yufang  and
      Kim, Gyuwan  and
      Kim, Young Jin  and
      Schuster, Tal  and
      Agrawal, Ameeta",
    booktitle = "Proceedings of the Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sustainlp-1.2/",
    doi = "10.18653/v1/2023.sustainlp-1.2",
    pages = "32--38"
}
```

Paper URL: https://arxiv.org/abs/2202.06133

---

## 9. Acknowledgements

This implementation builds on open-source ecosystems including:
- Hugging Face Transformers / Datasets
- Sentence Transformers
- SimCSE

