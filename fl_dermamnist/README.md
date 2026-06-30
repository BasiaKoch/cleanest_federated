# fl_dermamnist — FedAvg vs FedProx on DermaMNIST

Federated-learning pipeline used for the MPhil thesis comparing FedAvg
and FedProx on DermaMNIST (7-class skin lesion classification) under
statistical and computational-work heterogeneity.

Two runtimes coexist in this package: a pure-PyTorch reference loop
and a Flower-based simulation runtime. They are mathematically
equivalent at $\mu = 0$ and used for different chapters of the thesis
(see "Runtime roles" below).

## Layout

```
fl_dermamnist/
├── configs/base.yaml           # model.name selector
├── data/
│   ├── load.py                 # DermaMNIST loader (npz, 64→28 resize)
│   └── partition.py            # IID / engineered / specialist / Dirichlet / pathological
├── models/
│   └── dermmnist_cnn.py        # 4 conv blocks + GroupNorm + FC head
├── fl/
│   ├── local_train.py          # CE + gated proximal term (μ = 0 ⇒ FedAvg)
│   ├── aggregation.py          # size-weighted average of state dicts
│   ├── evaluation.py           # macro-F1, balanced accuracy, per-class F1
│   ├── provenance.py           # framework / git-commit canonicalisation
│   └── server_loop.py          # paired-fair FL run (pure-PyTorch)
├── fl_flower/                  # Flower-runtime client + strategy wiring
├── experiments/
│   ├── run_one.py                       # pure-PyTorch single run
│   ├── run_one_flower.py                # Flower single run
│   ├── run_one_fednova_flower.py        # Flower + FedNova arm
│   ├── run_centralised.py               # centralised (non-federated) reference
│   ├── verify_flower_equivalence.py     # μ = 0 equivalence smoke
│   └── compare_equivalence_full_scale.py# full pure-PyTorch ↔ Flower cross-check
├── analysis/                   # analyse_*.py — thesis tables + numeric summaries
├── figures/                    # plot_*.py — thesis figure generators (F_*.pdf)
├── common/                     # paths.py — repo/results/figure path resolver
├── results/                    # one directory per experiment (see below)
│   └── thesis_ready/           # data/ (aggregate tables) + figures/ (canonical F_*.pdf)
└── tests/                      # μ=0 equivalence, proximal-term, provenance guards
```

Local run/analysis helpers live in **`infra/local/commands.sh`** (one dispatcher
for sanity tests, single-seed training, and figure/table regeneration — *not* in
this package); conceptual-diagram generators live in **`docs/figure_generation/`**;
the full HPC sweep launchers live in **`infra/slurm/submit_*.sh`**.

## Runtime roles

- **Pure-PyTorch (`run_one.py` → `fl/server_loop.py`)** — reference
  loop. Used for the **engineered-partition headline** in the
  statistical-heterogeneity chapter (`results/headline/`). Provides a
  single-process sequential implementation with deterministic client
  iteration and bit-equivalent local objectives at $\mu = 0$.
- **Flower (`run_one_flower.py` → `fl_flower/`)** — simulation
  runtime. Used for (a) the **runtime-replication row** of the
  engineered headline (`results/flower_C0_baseline/`); (b) all
  **system-heterogeneity and partial-participation experiments**
  (`results/system_het_*`); (c) the **specialist falsification probe**
  (`results/specialist_partition/`); (d) the **μ sensitivity sweep**
  (`results/mu_sensitivity_flower/`); (e) the **FedNova comparator**
  (`results/system_het_random_fednova/`).

The pure-PyTorch and Flower runtimes give different effect sizes on the
same engineered partition; the thesis reports both side by side. The full
per-directory provenance ledger lives in **`docs/provenance/`**
(`result_traceability_matrix.csv` maps every claim → result → script → figure;
`numerical_verification_sheet.txt` re-derives each headline number from the raw
artefacts).

## Current experiment matrix

| Directory | Partition | Runtime | Seeds | Role in thesis |
|---|---|---|---|---|
| `centralised/` | pooled (no FL) | centralised PyTorch | 10 | Reference performance ceiling |
| `iid/` | `iid_7_clients` | pure-PyTorch | 10 | IID mechanism-null control |
| `headline/` | `balanced_paired_7_clients` | pure-PyTorch | 10 | **Engineered-partition primary headline** |
| `dirichlet_a01/` | `dirichlet_7_clients` (α = 0.1) | pure-PyTorch | 10 | Dirichlet non-IID alternative |
| `flower_C0_baseline/` | `balanced_paired_7_clients` | Flower 1.23.0 | 10 | Runtime replication of headline |
| `flower_C0_iid_baseline/` | `iid_7_clients` | Flower 1.23.0 | 10 | IID under Flower |
| `specialist_partition/` | `specialist_7_clients` | Flower 1.23.0 | 10 | Falsification probe (one class per client) |
| `system_het_fixed/` | engineered + fixed stragglers | Flower 1.23.0 | 10 | S1 — fixed-straggler condition |
| `system_het_random/` | engineered + random stragglers | Flower 1.23.0 | 10 | S2 — random-straggler condition |
| `system_het_iid_fixed/` | IID + fixed stragglers | Flower 1.23.0 | 10 | IID system-het control |
| `system_het_iid_random/` | IID + random stragglers | Flower 1.23.0 | 10 | IID system-het control |
| `system_het_random_asymmetric/` | engineered + asymmetric dropout | Flower 1.23.0 | 10 | Asymmetric-straggler protocol |
| `system_het_random_fednova/` | engineered + random stragglers + FedNova | Flower 1.23.0 | 10 | FedNova comparator |
| `system_het_partial_C0.5/` | engineered + $C = 0.5$ | Flower 1.23.0 | 10 | Partial-participation arm |
| `mu_sensitivity_flower/` | engineered | Flower 1.23.0 | 10 | μ ∈ {0.001, 0.01, 0.1, 1.0} sweep |
| `mu_sweep/` | engineered | pure-PyTorch | 3 | Older μ sweep (superseded by `mu_sensitivity_flower/`) |
| `arch_ablation_bn/` | engineered + BatchNorm | Flower 1.23.0 | 3 | **Archived — not used in thesis** (see directory README) |

Hyperparameters are shared across all federated experiments unless
noted: $R = 150$ rounds, $E = 20$ local epochs, SGD lr = 0.01,
momentum = 0.9, batch size 32, FedProx $\mu = 0.01$, cross-entropy
loss, no augmentation, 28×28 input resolution. Seeds:
{42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828}.

## Commands

All commands are also dispatched via `scripts/commands.sh <step>`.

### 1. Sanity test — FedProx(μ = 0) ≡ FedAvg

```bash
PYTHONPATH=. python -m pytest fl_dermamnist/tests/test_mu_zero_equals_fedavg.py -v
```

Verifies per-round metrics, final aggregated state-dict, and test
metrics at best-val checkpoint are numerically identical when $\mu = 0$.

### 2. One pure-PyTorch run

```bash
PYTHONPATH=. python -m fl_dermamnist.experiments.run_one \
    --algorithm fedavg --seed 42 --local-epochs 20 \
    --partition balanced_paired_7_clients

PYTHONPATH=. python -m fl_dermamnist.experiments.run_one \
    --algorithm fedprox --mu 0.01 --seed 42 --local-epochs 20 \
    --partition balanced_paired_7_clients
```

### 3. One Flower run

```bash
PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
    --algorithm fedprox --mu 0.01 --seed 42 --local-epochs 20 \
    --partition balanced_paired_7_clients \
    --num-rounds 150 --fraction-fit 1.0
```

Add `--device cuda` when on HPC. Add `--system-het random` for the S2
condition or `--system-het fixed` for S1.

### 4. Centralised reference

```bash
PYTHONPATH=. python -m fl_dermamnist.experiments.run_centralised \
    --seed 42 --num-epochs 50
```

### 5. Cross-runtime equivalence

```bash
PYTHONPATH=. python -m fl_dermamnist.experiments.verify_flower_equivalence \
    --seed 42
PYTHONPATH=. python -m fl_dermamnist.experiments.compare_equivalence_full_scale
```

### 6. Analysis (thesis-ready tables and figures)

Table generators live in **`fl_dermamnist/analysis/`** (`analyse_*.py`) and
figure generators in **`fl_dermamnist/figures/`** (`plot_*.py`); conceptual
diagrams live in `docs/figure_generation/`. They rebuild every thesis table and
figure from the **saved per-run results, without retraining**.

Canonical one-command path (from the repo root):

```bash
bash infra/local/commands.sh analyse     # rebuild all thesis tables + figures
```

Outputs land in `fl_dermamnist/results/thesis_ready/{data,figures}/`; the report
bundle (`report/supporting/`) carries flat copies of the `F_*.pdf` figures. The
exact section → script → figure mapping is the navigation table in the top-level
`README.md`, and every reported number is traced in `docs/provenance/`.

## Statistical claims policy

The headline engineered comparison reports:

- mean $\pm$ standard deviation across the 10 paired seeds (ddof = 1),
- per-seed Δ = FedProx − FedAvg (paired),
- paired-$t$ 95% confidence interval on the within-pair Δ,
- two-sided paired Wilcoxon signed-rank $p$-value,
- count of paired seeds on which FedProx wins.

For per-class tables, the family of 7 per-class tests is Holm-adjusted
when reported as inferential claims; otherwise per-class Δ values are
reported descriptively. The partial-participation comparison at
$C = 0.5$ is **not strictly paired** (the FedAvg and FedProx arms
desynchronise on 5 of 10 seeds because FedProx consumes one extra
random number per round) — the outcome-level comparison there is
reported as supportive evidence only; the load-bearing claim in that
section is the within-arm update-norm comparison.

## Spec compliance cross-reference

| Requirement | Location |
|---|---|
| FedAvg local objective: plain CE | `fl/local_train.py` (gated `if proximal_mu > 0`) |
| FedProx local objective: CE + $(\mu / 2) \cdot \lVert w - w_g \rVert^2$ | `fl/local_train.py` |
| `global_weights_frozen` is detached, not aliased | `fl/server_loop.py` `freeze_global_weights` (called once per round); `tests/test_fedprox_proximal_term.py:test_global_weights_frozen_is_not_aliased` |
| $\mu = 0$ ≡ FedAvg numerically | `tests/test_mu_zero_equals_fedavg.py` |
| Aggregation is size-weighted | `fl/aggregation.py:weighted_average_state_dicts` |
| Paired runs: same init, partition, dataloader RNG | `fl/server_loop.py` (`dataloader_generator_seed(seed, round, client)`) |
| Test evaluated only at best-val checkpoint | `fl/server_loop.py` (argmax of `val_macro_f1`, evaluated once after training) |
| Pure-PyTorch ↔ Flower equivalence at $\mu = 0$ | `experiments/verify_flower_equivalence.py`, `experiments/compare_equivalence_full_scale.py`, `tests/test_framework_provenance.py` |
| Provenance ledger | `docs/provenance/result_traceability_matrix.csv` + `numerical_verification_sheet.txt` |
