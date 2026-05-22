# headline/ — provenance notice

**Legacy pure-PyTorch only.** Do not mix with Flower outputs in this directory.

## What this directory contains

20 `test_at_best_*.json` files plus matching `history_*.csv` for FedAvg + FedProx,
10 paired seeds {42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828},
at $E=20$, $R=150$, $\mu=0.01$, partition `balanced_paired_7_clients`. Every JSON
carries:

- `framework: "pure-pytorch"`
- `runner_script: "run_one.py"`
- `provenance_note: "backstamped-2026-05-18"`

These files were produced by the project's pure-PyTorch reference loop
(`mnist_dermnist/experiments/run_one.py` → `mnist_dermnist/fl/server_loop.py`)
*before* the Flower runtime was introduced. They are preserved as the
**cross-runtime validation reference** consumed by
`mnist_dermnist/experiments/compare_equivalence_full_scale.py`.

## What this directory is NOT

This is not the canonical Flower-runtime headline. That lives in
`../flower_C0_baseline/` and carries `framework: "flower-simulation"`,
`framework_version: "1.23.0"` (or whatever the HPC venv has). The thesis
text after the 2026-05-22 relabel patch describes the pure-PyTorch
implementation as the historical reference; new claims should cite the
Flower data.

## How to tell legacy from canonical at a glance

```bash
python -c "
import json
print(json.load(open('mnist_dermnist/results/headline/test_at_best_fedavg_mu0.0_E20_s42.json'))['framework'])
# expected: 'pure-pytorch'
print(json.load(open('mnist_dermnist/results/flower_C0_baseline/test_at_best_fedavg_mu0.0_E20_s42.json'))['framework'])
# expected: 'flower-simulation'
"
```

The `mnist_dermnist.fl.provenance.canonicalise_framework` helper maps both to
canonical labels; the unit test
`tests/test_framework_provenance.py:test_all_20_headline_jsons_canonicalise_to_pure_pytorch`
reads this directory's JSONs every CI run and asserts they all resolve to
`"pure-pytorch"`. If that test fails, this directory has been contaminated.

## Why not rename the directory

A safer-than-rename plan: documented separation via this README. Renaming
`headline/` to `legacy_pure_pytorch/` would break the hard-coded default
paths in 11+ analysis scripts (`tables.py`, `plots.py`,
`analyse_extra_statistics.py`, `analyse_per_client.py`,
`analyse_best_vs_last.py`, `generate_curves.py`,
`analyse_worst_case_per_class.py`, `plot_per_class_delta.py`,
`analyse_communication_metrics.py`, `generate_thesis_figures_10_12.py`,
`analyse_confusion_matrices.py`, `check_results.sh`, `analyse_all.sh`).
Some of those defaults are documentary and some are functional — patching
all of them in one atomic commit is the only safe way to rename. Until
that's done, the directory keeps its historical name and the README does
the disambiguation work.
