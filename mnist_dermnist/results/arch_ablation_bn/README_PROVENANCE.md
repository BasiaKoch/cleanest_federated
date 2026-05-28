# arch_ablation_bn/ — archived ablation, NOT used in thesis

**Status:** archived. This directory contains an incomplete
architecture ablation (BatchNorm variant of `DermMNISTCNN`) that is
**not used in the main thesis**.

## What this directory contains

12 history CSVs and 12 client-update-norm CSVs plus 6 `test_at_best_*.json`
files, covering 3 seeds {42, 123, 456} × {FedAvg, FedProx} = 6 paired
runs at Flower runtime, $E = 20$, $R = 150$, partition
`balanced_paired_7_clients`, run with the BN variant
`DermMNISTCNN_BN` instead of the headline GroupNorm model
`DermMNISTCNN`. Each JSON carries:

- `framework: "flower-simulation"`
- `framework_version: "1.23.0"`
- `runner_script: "run_one_flower.py"`
- `git_commit: 0b7f69ae3742dd90ef60e57fa7c2656a7ac94853`
- model identifier in filename suffix `arch-bn`

## Why this is not in the thesis

1. **Incomplete (n = 3 seeds)** versus the 10-seed standard used
   elsewhere in the thesis; the across-seed standard deviations are
   not directly comparable to the headline tables.
2. **BatchNorm running statistics are not aggregated under federated
   averaging.** Each client's BN buffers (`running_mean`, `running_var`)
   remain client-local after `weighted_average_state_dicts`, which
   silently breaks the federated assumption — the global model
   inherits whichever client's BN buffers came last in dictionary
   iteration order. This is a known FL-with-BN pathology (cf. FedBN,
   Li et al. 2021) and the headline model uses GroupNorm precisely to
   avoid it.
3. The thesis architecture is fixed at `DermMNISTCNN` (GroupNorm); no
   architecture-ablation chapter or appendix discusses BN variants.

## Retention rationale

The data is retained as a record of the BN experiment and as a future
starting point if a FedBN-style aggregation rule is ever introduced;
deletion would lose 6 × ~150-round trajectories that took non-trivial
HPC time to produce. The directory is kept but flagged as archived via
this README and via the `PROVENANCE_AUDIT.md` ledger at
`mnist_dermnist/results/PROVENANCE_AUDIT.md`.

## Do not include in analysis pipelines

The scripts `scripts/check_results.sh` and `scripts/analyse_all.sh` do
not list this directory in their sweep registries. If a future
analyser starts auto-discovering result directories, it should treat
this directory as archived (skip, or warn that it is BN-variant data).
