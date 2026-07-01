#!/bin/bash
# All commands for the fl_dermamnist FedAvg/FedProx thesis pipeline.
# Run from the repository root: /Users/basiakoch/cleanest_federated
#
# Dispatcher usage:
#   bash infra/local/commands.sh <step>
#
# All commands prepend PYTHONPATH=. because tests/scripts use absolute imports.
set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Sanity test - FedProx(μ=0) == FedAvg
# ----------------------------------------------------------------------------
sanity_test() {
  PYTHONPATH=. python -m pytest \
    fl_dermamnist/tests/test_mu_zero_equals_fedavg.py \
    fl_dermamnist/tests/test_fedprox_proximal_term.py \
    fl_dermamnist/tests/test_framework_provenance.py \
    -v
}

# ----------------------------------------------------------------------------
# 2. One pure-PyTorch run (engineered partition, seed=42, E=20)
# ----------------------------------------------------------------------------
run_purepy_fedavg() {
  PYTHONPATH=. python -m fl_dermamnist.experiments.run_one \
    --algorithm fedavg --seed 42 --local-epochs 20 \
    --partition balanced_paired_7_clients \
    --num-rounds 150
}

run_purepy_fedprox() {
  PYTHONPATH=. python -m fl_dermamnist.experiments.run_one \
    --algorithm fedprox --mu 0.01 --seed 42 --local-epochs 20 \
    --partition balanced_paired_7_clients \
    --num-rounds 150
}

# ----------------------------------------------------------------------------
# 3. One Flower run (engineered partition, seed=42, E=20)
# ----------------------------------------------------------------------------
run_flower_fedavg() {
  PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
    --algorithm fedavg --seed 42 --local-epochs 20 \
    --partition balanced_paired_7_clients \
    --num-rounds 150 --fraction-fit 1.0
}

run_flower_fedprox() {
  PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
    --algorithm fedprox --mu 0.01 --seed 42 --local-epochs 20 \
    --partition balanced_paired_7_clients \
    --num-rounds 150 --fraction-fit 1.0
}

# ----------------------------------------------------------------------------
# 4. Centralised reference (single seed)
# ----------------------------------------------------------------------------
run_centralised() {
  PYTHONPATH=. python -m fl_dermamnist.experiments.run_centralised \
    --seed 42 --num-epochs 50
}

# ----------------------------------------------------------------------------
# 5. Cross-runtime equivalence (pure-PyTorch ↔ Flower at μ=0)
# ----------------------------------------------------------------------------
verify_equivalence() {
  PYTHONPATH=. python -m fl_dermamnist.experiments.verify_flower_equivalence \
    --seed 42
}

compare_equivalence_full_scale() {
  PYTHONPATH=. python -m fl_dermamnist.experiments.compare_equivalence_full_scale
}

# ----------------------------------------------------------------------------
# 6. Thesis-ready analysis pipeline
# ----------------------------------------------------------------------------
analyse_thesis() {
  bash infra/local/analyse_all.sh
}

# ----------------------------------------------------------------------------
# 7. Inspect overall run / queue status
# ----------------------------------------------------------------------------
check_status() {
  bash infra/local/check_results.sh
}

# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------
case "${1:-}" in
  sanity)              sanity_test ;;
  purepy-fedavg)       run_purepy_fedavg ;;
  purepy-fedprox)      run_purepy_fedprox ;;
  flower-fedavg)       run_flower_fedavg ;;
  flower-fedprox)      run_flower_fedprox ;;
  centralised)         run_centralised ;;
  verify-equivalence)  verify_equivalence ;;
  equivalence-full)    compare_equivalence_full_scale ;;
  analyse)             analyse_thesis ;;
  status)              check_status ;;
  *)
    cat <<EOF
Usage: bash $0 <step>

Steps:
  sanity              run μ=0 ≡ FedAvg + proximal-term + provenance unit tests
  purepy-fedavg       one pure-PyTorch FedAvg run, seed 42, engineered partition
  purepy-fedprox      one pure-PyTorch FedProx run (μ=0.01), seed 42, engineered partition
  flower-fedavg       one Flower FedAvg run, seed 42, engineered partition
  flower-fedprox      one Flower FedProx run (μ=0.01), seed 42, engineered partition
  centralised         one centralised reference run, seed 42
  verify-equivalence  pure-PyTorch ↔ Flower smoke test at μ=0 (single seed)
  equivalence-full    full-scale pure-PyTorch ↔ Flower cross-runtime audit
  analyse             run the thesis-ready analysis pipeline (tables + figures)
  status              show per-experiment-directory completion status

The actual thesis experiments were run as 10-seed sweeps on HPC; this
script's one-seed entry points are for local smoke checks. Per-experiment
hyperparameters, seeds, and partitions are documented in
fl_dermamnist/results/PROVENANCE_AUDIT.md and the top-level README.md.
EOF
    exit 1
    ;;
esac
