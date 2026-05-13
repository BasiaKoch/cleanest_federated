# mnist_dermnist — Headline FedAvg vs FedProx on DermMNIST

Pure-PyTorch (no Flower/Ray) federated-learning pipeline. Designed for
rigorous paired comparison: given a seed, both algorithms see identical
initial weights, partition, sampling schedule, and minibatch order.

## Layout

```
mnist_dermnist/
├── configs/base.yaml           # model.name selector
├── data/
│   ├── load.py                 # DermMNIST loader (npz, 28x28 resize)
│   └── partition.py            # simple_pathological_3 / medical_skew_7
├── models/
│   └── dermmnist_cnn.py        # 4 conv blocks + GroupNorm + FC head
├── fl/
│   ├── local_train.py          # CE + gated proximal term
│   ├── aggregation.py          # weighted average of state dicts
│   ├── evaluation.py           # macro-F1, bACC, per-class F1
│   └── server_loop.py          # paired-fair FL run
├── experiments/
│   ├── run_one.py              # single (algo, mu, seed, E) run
│   ├── run_headline_sweep.py   # 10 seeds × {FedAvg, FedProx} × E=20
│   └── run_e_sweep.py          # optional E sweep
├── analysis/
│   ├── tables.py               # paired stats: Wilcoxon, rank-biserial
│   └── plots.py                # 4 plots
├── scripts/commands.sh         # all commands in one place
└── tests/                      # 38 unit tests
```

## Commands

All commands are wrapped in `scripts/commands.sh` so you can run them as
`bash mnist_dermnist/scripts/commands.sh <step>`. The raw form is below.

### 1. Sanity test — FedProx(μ=0) ≡ FedAvg

```bash
PYTHONPATH=. python -m pytest mnist_dermnist/tests/test_mu_zero_equals_fedavg.py -v
```

The 7 tests verify per-round metrics, final state dicts, and best-val test
metrics are numerically identical under paired seeds.

### 2. One FedAvg run (seed=42, E=20)

```bash
PYTHONPATH=. python -m mnist_dermnist.experiments.run_one \
    --algorithm fedavg --seed 42 --local-epochs 20
```

### 3. One FedProx run (μ=0.1, seed=42, E=20)

```bash
PYTHONPATH=. python -m mnist_dermnist.experiments.run_one \
    --algorithm fedprox --mu 0.1 --seed 42 --local-epochs 20
```

### 4. Full headline sweep

```bash
PYTHONPATH=. python -m mnist_dermnist.experiments.run_headline_sweep \
    --local-epochs 20 --num-rounds 150 --lr 0.01 --batch-size 32 \
    --partition medical_skew_7_clients \
    --out-dir mnist_dermnist/results/headline
```

- algorithms: fedavg, fedprox(μ=0.1)
- seeds: 42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828
- partition: medical_skew_7_clients
- model: DermMNISTCNN
- loss: cross-entropy (no class weighting)
- optimizer: SGD lr=0.01 momentum=0.9
- batch_size: 32 (capped per-client by client_size)
- rounds: 150

Total: 20 runs. Skips already-completed runs (`test_at_best_*.json` exists).

Add `--device cuda` when on HPC. Add `--dry-run` to print commands without
executing.

### 5. Optional E sweep

```bash
PYTHONPATH=. python -m mnist_dermnist.experiments.run_e_sweep \
    --local-epochs 1 5 10 20 40 --seeds 42 123 456 \
    --num-rounds 100 --out-dir mnist_dermnist/results/e_sweep
```

### 6. Analysis

**Partition heatmap data** (run once per partition+seed):

```bash
PYTHONPATH=. python -m mnist_dermnist.data.partition \
    --mode medical_skew_7_clients --seed 42 \
    --out mnist_dermnist/results/partitions
```

**Tables** (per-seed final test, paired Wilcoxon, rank-biserial, per-class):

```bash
PYTHONPATH=. python -m mnist_dermnist.analysis.tables \
    --results-dir mnist_dermnist/results/headline --E 20
```

Outputs in `<results-dir>/analysis/`:
- `final_test_table.csv` — per-seed FedAvg, FedProx, Δ
- `paired_stats.json`    — mean±std, Wilcoxon p (two-sided + greater), rank-biserial
- `per_class_diff.csv`   — class-by-class FedProx − FedAvg

**Plots** (4 panels):

```bash
PYTHONPATH=. python -m mnist_dermnist.analysis.plots \
    --results-dir mnist_dermnist/results/headline --E 20 \
    --partition-counts mnist_dermnist/results/partitions/partition_medical_skew_7_clients_seed42_counts.csv
```

Generates `headline_E20.{png,pdf}`:
- (A) Validation macro-F1 vs round (mean ± SEM)
- (B) Training loss vs round (mean ± SEM)
- (C) Per-class test F1 bar plot (mean ± std)
- (D) Client-class distribution heatmap

## Statistical claims policy

The analysis script applies the **paired Wilcoxon signed-rank test** on
seed-level final test macro-F1 differences. With 10 paired seeds the test
can detect a real effect; with fewer than 5 it cannot reach conventional
significance even if FedProx wins every pair.

Verdicts printed by `tables.py`:

| Condition | Verdict |
|---|---|
| `n_pairs < 5` | insufficient seeds — descriptive only |
| `p < 0.05` AND `mean_diff > 0` | FedProx significantly outperforms FedAvg |
| `p < 0.05` AND `mean_diff < 0` | FedAvg significantly outperforms FedProx |
| `p ≥ 0.05` | no statistically significant difference at α=0.05 |

The script warns loudly when paired seeds are missing on either side.

## Spec compliance (cross-reference)

| Requirement | Where |
|---|---|
| FedAvg local objective: plain CE | `fl/local_train.py` (gated `if proximal_mu > 0:`) |
| FedProx local objective: CE + (μ/2)·Σ‖w−w_g‖² | `fl/local_train.py` |
| `global_weights_frozen` is detached deep copy, unchanged during training | `fl/server_loop.py` (`freeze_global_weights` called ONCE per round) |
| μ=0 ≡ FedAvg numerically | `tests/test_mu_zero_equals_fedavg.py` (7 tests, all pass) |
| Paired runs: same init, partition, sampling, batch order | `fl/server_loop.py` (seeded RNGs at every layer) |
| Per-round metrics in CSV | `fl/server_loop.py` returns DataFrame with required cols |
| Test only at best-val checkpoint | `fl/server_loop.py` tracks `best_state_dict` by `val_macro_f1` |
| Paired Wilcoxon + effect size | `analysis/tables.py` |
| Don't claim FedProx wins unless stats support it | `tables.py` VERDICT section |
| Warn when seeds missing | `tables.py` warns loudly via `warnings.warn` |
