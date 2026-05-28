# headline/ — provenance notice

**Pure-PyTorch reference loop. Primary analysis for the engineered-partition headline.**
Do not mix outputs from this directory with Flower outputs.

## What this directory contains

20 `test_at_best_*.json` files plus matching `history_*.csv` for FedAvg + FedProx,
10 paired seeds {42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828},
at $E=20$, $R=150$, $\mu=0.01$, partition `balanced_paired_7_clients`. Every JSON
carries:

- `framework: "pure-pytorch"`
- `runner_script: "run_one.py"`
- `provenance_note: "backstamped-2026-05-18"`

These files were produced by the project's pure-PyTorch reference loop
(`mnist_dermnist/experiments/run_one.py` → `mnist_dermnist/fl/server_loop.py`),
a single-process sequential implementation with deterministic client iteration,
in-memory parameter passing, and explicit size-weighted aggregation.

## Role in the thesis

This directory is the **primary headline** for the engineered-partition
FedAvg-vs-FedProx comparison reported in the statistical-heterogeneity chapter
(`clean_federated_overleaf.tex`, Section `sec:engineered-headline`,
Table `tab:engineered-results` first row). On these data the within-pair
$\Delta$ in test macro-F1 is $+0.0267$ ($9/10$ paired-seed wins, paired
Wilcoxon $p = 0.020$, paired-$t$ $95\%$ CI on $\Delta$: $[+0.002, +0.052]$).
The per-class decomposition (`tab:engineered-per-class`), the per-class
validation-trajectory figure (`fig:engineered-per-class`), and the
mechanism-null IID and Dirichlet contrasts are all anchored on this
pure-PyTorch reference loop.

## Relationship to the Flower runtime

The Flower runtime replication of the same engineered partition lives in
`../flower_C0_baseline/` (`framework: "flower-simulation"`). It is reported
as the **second row** of Table `tab:engineered-results` and is the
runtime-sensitivity check, not the primary analysis. On Flower the within-
pair $\Delta$ is $+0.0069$ ($7/10$ wins, Wilcoxon $p = 0.43$, $95\%$ CI on
$\Delta$: $[-0.011, +0.024]$). The direction of effect replicates across
runtimes; the magnitude attenuates by roughly $4\times$ and the Flower
result does not clear $\alpha = 0.05$. The thesis reports both numbers
side by side and unpacks the runtime gap in Section
`sec:runtime-sensitivity`; it does not present the larger pure-PyTorch
effect in isolation.

Flower is also the canonical runtime for the **robustness and
system-heterogeneity** chapters: the fixed- and random-straggler
conditions (`../system_het_fixed/`, `../system_het_random/`), the
asymmetric-straggler protocol (`../system_het_random_asymmetric/`),
the FedNova comparison (`../system_het_random_fednova/`), the partial-
participation arm (`../system_het_partial_C0.5/`), and the $\mu$
sensitivity sweep (`../mu_sensitivity_flower/`) are all Flower-runtime.
Comparisons within those chapters stay within the Flower runtime so that
runtime is not confounded with the manipulation.

Cross-runtime mathematical equivalence at $\mu = 0$ is verified by
`mnist_dermnist/experiments/compare_equivalence_full_scale.py`, which
consumes this directory as its pure-PyTorch reference.

## How to tell pure-PyTorch from Flower at a glance

```bash
python -c "
import json
print(json.load(open('mnist_dermnist/results/headline/test_at_best_fedavg_mu0.0_E20_s42.json'))['framework'])
# expected: 'pure-pytorch'
print(json.load(open('mnist_dermnist/results/flower_C0_baseline/test_at_best_fedavg_mu0.0_E20_s42.json'))['framework'])
# expected: 'flower-simulation'
"
```

The `mnist_dermnist.fl.provenance.canonicalise_framework` helper maps both
to canonical labels; the unit test
`tests/test_framework_provenance.py:test_all_20_headline_jsons_canonicalise_to_pure_pytorch`
reads this directory's JSONs every CI run and asserts they all resolve to
`"pure-pytorch"`. If that test fails, this directory has been contaminated.

## Why the directory keeps its name

Renaming `headline/` to e.g. `pure_pytorch_engineered/` would break the
hard-coded default paths in 11+ analysis and plotting scripts
(`tables.py`, `plots.py`, `analyse_extra_statistics.py`,
`analyse_per_client.py`, `analyse_best_vs_last.py`, `generate_curves.py`,
`analyse_worst_case_per_class.py`, `plot_per_class_delta.py`,
`analyse_communication_metrics.py`, `generate_thesis_figures_10_12.py`,
`analyse_confusion_matrices.py`, `check_results.sh`, `analyse_all.sh`).
The directory name is historical; the framing is set by this README and
by the thesis text, which jointly establish that the data here is the
pure-PyTorch reference primary analysis for the engineered partition and
that Flower in `../flower_C0_baseline/` is the runtime replication.
