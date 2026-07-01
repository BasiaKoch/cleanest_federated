# Preserving Rare-Class Signal in Federated Learning under Statistical and System Heterogeneity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This repository is associated with the submission of the Research Project for the **MPhil in Data Intensive Science** at the **University of Cambridge**. The coursework instructions can be found under [Instructions.md](Instructions.md). The associated project report can be found under [Project Report](report/report.pdf) and the associated executive summary under [Executive Summary](report/executive_summary.pdf).

The project studies **federated learning (FL) on DermaMNIST** — a seven-class dermatology benchmark (28×28) that is severely class-imbalanced (melanocytic nevi ≈ 67 % of the data, while the clinically urgent lesions are ≈ 1–11 %). The primary objective is not to decide *which* federated optimiser is best, but to map **under which conditions** the standard aggregator **FedAvg** and the heterogeneity-oriented methods **FedProx** and **FedNova** preserve or destroy the rare-class signal. The contribution is a **regime map**: performance is regime-dependent, and rare-class F1 is preserved precisely when the rare client's signal reaches the global model with usable weight. Full context, methods, and findings are in the report.

## Table of Contents
- [Data Availability](#data-availability)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Reproducing the Report](#reproducing-the-report)
- [Support](#support)
- [License](#license)
- [Documentation](#documentation)
- [Note on the Use of Auto-generation Tools](#note-on-the-use-of-auto-generation-tools)
- [Author and Acknowledgment](#author-and-acknowledgment)

## Data Availability

The project uses **DermaMNIST** from [MedMNIST v2](https://medmnist.com/), derived from the HAM10000 dermatoscopic collection. The dataset is **not versioned with the code**: it is read locally from `dermamnist_64.npz` at the repository root (git-ignored, ~100 MB) and the loader resizes it to the 28×28 resolution used throughout. It is freely re-obtainable through the pinned `medmnist` package (installed with the requirements below). HAM10000 is released under **CC BY-NC 4.0**.

## Installation

To get started with the code base, follow these steps.

### Requirements
- Python **3.14** (tested).
- `pip` and `venv`.
- All Python dependencies are pinned in [`requirements.txt`](requirements.txt).

### Setup

1. **Clone the repository:**

    ```
    $ git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/bk489
    ```

2. **Navigate to the project directory:**

    ```
    $ cd /full/path/to/bk489
    ```

    and replace `/full/path/to/` with the directory on your machine where the repository lives.

3. **Create the environment and install:**

    ```
    $ python -m venv .venv && source .venv/bin/activate
    $ pip install -r requirements.txt
    ```

    (or `pip install -e .` to install the `fl_dermamnist` package in editable mode).

4. **Add the dataset:** place `dermamnist_64.npz` at the repository root (see `fl_dermamnist/data/load.py`).

## Usage

All entrypoints run as modules from the repository root; the `infra/local/commands.sh` dispatcher wraps the common ones.

### 1. Sanity checks (implementation guards)

Verify FedProx(μ = 0) ≡ FedAvg, the proximal term, the FedNova normaliser, and provenance fields before trusting any result:

```
$ bash infra/local/commands.sh sanity
```

### 2. A single training run (local, one seed)

```
$ bash infra/local/commands.sh flower-fedprox     # Flower simulation runtime
$ bash infra/local/commands.sh purepy-fedavg      # pure-PyTorch reference loop
```

or call an entrypoint directly, for example:

```
$ PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
      --algorithm fedprox --mu 0.01 --seed 42 --local-epochs 20 \
      --partition balanced_paired_7_clients --num-rounds 150
```

The training entrypoints live in `fl_dermamnist/experiments/`: `run_one.py` (pure-PyTorch FedAvg/FedProx), `run_one_flower.py` (Flower FedAvg/FedProx, with stragglers and partial participation), `run_one_fednova_flower.py` (FedNova), and `run_centralised.py` (non-federated ceiling), plus fine-tuning and cross-runtime equivalence checks. Each run writes its full configuration and metrics to `test_at_best_*.json`, per-round trajectories to `history_*.csv`, and test predictions to `test_predictions_*.npz`, into one directory per experiment under `fl_dermamnist/results/`.

### 3. Regenerate the report's tables and figures (no retraining)

```
$ bash infra/local/commands.sh analyse
```

This rebuilds every thesis table and figure from the saved per-run results, writing to `fl_dermamnist/results/thesis_ready/{data,figures}/`.

### 4. Full hyperparameter sweeps (HPC)

The complete sweeps were run on HPC via the launchers in `infra/slurm/submit_*.sh`; they are costly to re-run, so the saved outputs above are the intended reproduction path.

## Repository Structure

```
bk489/
├── README.md                 ← this file
├── Instructions.md           ← coursework instructions
├── LICENSE                   ← MIT
├── requirements.txt          ← pinned dependencies (Python 3.14)
├── pyproject.toml            ← installable package (pip install -e .)
├── dermamnist_64.npz         ← dataset (git-ignored, ~100 MB, re-downloadable)
├── fl_dermamnist/            ← the source package
│   ├── data/                 ← DermaMNIST loader + client partitions
│   ├── models/               ← the GroupNorm CNN (+ BatchNorm ablation)
│   ├── fl/                   ← pure-PyTorch FL core (local train, aggregation, eval, stragglers)
│   ├── fl_flower/            ← Flower simulation runtime (clients + FedNova/straggler strategies)
│   ├── experiments/          ← training entrypoints (run_*.py)
│   ├── analysis/             ← analyse_*.py — thesis tables + numeric summaries
│   ├── figures/              ← plot_*.py — thesis figure generators
│   ├── tests/                ← implementation guards (μ=0≡FedAvg, proximal term, provenance)
│   └── results/              ← one directory per experiment; thesis_ready/ holds curated data + figures
├── infra/                    ← local/ (commands.sh dispatcher) + slurm/ (HPC launchers)
├── report/                   ← report.pdf + executive_summary.pdf; supporting/ holds the LaTeX sources
└── docs/                     ← provenance/ (traceability matrix + verification ledger) + figure_generation/
```

For a detailed, per-module description, the full experiment matrix, and a spec-compliance cross-reference, see [`fl_dermamnist/README.md`](fl_dermamnist/README.md).

## Reproducing the Report

- **Compile the report.** The LaTeX sources are in [`report/supporting/`](report/supporting) — a flat, self-contained bundle. Compile `main.tex` (the thesis) and `executive_summary.tex` (the standalone summary) with **pdfLaTeX + BibTeX** to produce `report/report.pdf` and `report/executive_summary.pdf`.
- **Rebuild figures and tables** from saved results with `bash infra/local/commands.sh analyse` (CPU, minutes; no retraining).
- **Traceability.** Every reported number maps to a source artefact through `docs/provenance/result_traceability_matrix.csv` (claim → experiment → script → figure) and is re-derived in `docs/provenance/numerical_verification_sheet.txt`.
- **Seed tiers.** Headline comparisons use **n = 10** paired seeds (paired wins / Wilcoxon where applicable); mechanism probes use **n = 3** matched seeds (directional only); the L0–L4 heterogeneity ladder is a single-seed pilot. Seed counts vary because HPC access was limited, and each is reported in the report.

## Support
For any questions, feedback, or assistance regarding this submission, please contact the author at [bk489@cam.ac.uk](mailto:bk489@cam.ac.uk).

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. The DermaMNIST data is not included and is redistributed by MedMNIST v2 from HAM10000 under **CC BY-NC 4.0**; obtain it via the pinned `medmnist` package (see [Installation](#installation)).

## Documentation
- **Detailed code documentation** — package layout, runtime roles, the experiment matrix, and a spec-compliance cross-reference: [`fl_dermamnist/README.md`](fl_dermamnist/README.md).
- **Provenance and verification** — `docs/provenance/` (`result_traceability_matrix.csv`, `numerical_verification_sheet.txt`).
- **Curated outputs** — `fl_dermamnist/results/thesis_ready/` holds the tables and figures the report draws from.

## Note on the Use of Auto-generation Tools
**Claude Code** (Anthropic's CLI, using the **Claude Opus 4.8** model) was used for assistance with coding, debugging, running experiments, repository organisation, and preparing the report and this documentation. **ChatGPT** (OpenAI) was used for organisation, planning, and structuring support. No results were generated by these tools: all experiments, numerical results, their interpretation, and the scientific claims are the author's own work and remain the author's responsibility, and AI-assisted code and outputs were reviewed by the author. This statement should be read alongside, and complies with, the relevant course policy on the use of auto-generation tools.

## Author and Acknowledgment
This project is submitted by **Barbara Koch**, MPhil in Data Intensive Science, University of Cambridge (Peterhouse), under the supervision of **Dr Joshua Kaggie**.

1st July 2026
