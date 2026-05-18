# HPC submission instructions — System Heterogeneity sweep

These are the exact commands to run on the CSD3 login node when HPC
resources are available again, plus a verification checklist.

## Prerequisite: push the new code from laptop

```bash
cd /Users/basiakoch/cleanest_federated
git add mnist_dermnist/fl/system_het.py \
        mnist_dermnist/fl/server_loop.py \
        mnist_dermnist/experiments/run_one.py \
        mnist_dermnist/scripts/submit_system_het.sh \
        mnist_dermnist/scripts/slurm_template_system_het.sh \
        mnist_dermnist/results/thesis_ready_system_het/
git commit -m "Add system-heterogeneity experiment: code, scripts, thesis_ready folder"
git push
```

## On HPC: pull and verify

```bash
cd /home/bk489/federated_clean/cleanest_federated
git pull
source /home/bk489/federated_clean/.venv/bin/activate

# Verify the new flag is wired:
PYTHONPATH=. python -m mnist_dermnist.experiments.run_one --help 2>&1 | grep -A1 system-het
# Expected: shows --system-het-mode {uniform,fixed_stragglers,random_stragglers}
```

## CPU smoke test (~1 minute on login node, optional but recommended)

Verify the new code path works before submitting 40 SLURM jobs:

```bash
PYTHONPATH=. python -m mnist_dermnist.experiments.run_one \
  --algorithm fedprox --mu 0.01 --seed 42 \
  --local-epochs 2 --num-rounds 2 \
  --partition balanced_paired_7_clients \
  --device cpu \
  --system-het-mode fixed_stragglers --straggler-epochs 1 \
  --npz-path /home/bk489/federated_clean/cleanest_federated/dermamnist_64.npz \
  --out-dir /tmp/sh_smoke_fixed

PYTHONPATH=. python -m mnist_dermnist.experiments.run_one \
  --algorithm fedprox --mu 0.01 --seed 42 \
  --local-epochs 2 --num-rounds 2 \
  --partition balanced_paired_7_clients \
  --device cpu \
  --system-het-mode random_stragglers --straggler-fraction 0.5 \
  --npz-path /home/bk489/federated_clean/cleanest_federated/dermamnist_64.npz \
  --out-dir /tmp/sh_smoke_random
```

Both should complete without traceback and produce filenames tagged
`sh-fixed_stragglers` or `sh-random_stragglers` respectively.

## Submit the full sweep (40 jobs ~40 GPU-hours)

```bash
bash mnist_dermnist/scripts/submit_system_het.sh
```

Verify the count:

```bash
squeue -u $USER | tail -n +2 | wc -l
# Expected: 40 (plus any jobs already queued from IID/Dirichlet sweep)
```

## Monitor

```bash
echo "Fixed-stragglers:  $(ls mnist_dermnist/results/system_het_fixed/test_at_best_*.json 2>/dev/null | wc -l) / 20"
echo "Random-stragglers: $(ls mnist_dermnist/results/system_het_random/test_at_best_*.json 2>/dev/null | wc -l) / 20"
echo "Queued/running:    $(squeue -u $USER -h | wc -l)"
```

## When all 40 finish — analyse

```bash
# Aggregate using the dedicated system-het analysis script
PYTHONPATH=. python mnist_dermnist/results/thesis_ready_system_het/scripts/analyse_system_het.py
```

This produces:
- `thesis_ready_system_het/data/summary_statistics.json`
- `thesis_ready_system_het/data/per_seed_results.csv`

And prints the headline results table to stdout, including:
- Per-condition mean ± SD for FedAvg and FedProx
- Within-condition Δ and Wilcoxon p (H1)
- Between-condition Δ (vs baseline) and Wilcoxon p (H2)
- Rank-biserial effect sizes
- Straggler-tolerance ratios (ρ_FedAvg, ρ_FedProx)

Then update the `\TODOhpcSH{...}` placeholders in
`thesis_ready_system_het/writing/03_overleaf_ready_system_het.tex` and the
section is complete.

## Pre-registration

The hypotheses H1 and H2 are pre-registered in
`thesis_ready_system_het/conclusions/hypothesis_and_expected_outcomes.md`
**before** the HPC sweep results are seen. Refer to that document for the
falsification criteria and the interpretive framework.
