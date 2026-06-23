#!/bin/bash
# Clean 10-seed FedProx mu-sensitivity sweep on the engineered partition.
#
# Replaces results/mu_sweep/ (3-seed pilot, no update-norm logging,
# no git_commit provenance). Writes to a fresh directory so the new
# sweep cannot be silently mixed with the pilot.
#
# Design:
#   - 5 mu values  : 0, 0.001, 0.01, 0.1, 1.0
#   - 10 seeds     : 42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828
#   - 50 runs total. mu=0 routes via --algorithm fedavg (the runner forces
#     mu=0.0 in that branch and the proximal-term code path is gated off).
#   - All jobs pass --log-update-norms so the mechanism CSVs are written.
#   - All other hyperparameters match the engineered C0 baseline exactly:
#       balanced_paired_7_clients, E=20, R=150, C=1.0, SGD lr=0.01 m=0.9,
#       weight_decay=0, batch_size=32, loss=ce, model=DermMNISTCNN(GroupNorm).
#
# Provenance:
#   - run from a clean working tree (commit hash will be recorded in every
#     test_at_best_*.json under `git_commit`).
#   - dispatch this script after smoke-testing one job locally and
#     confirming the update-norm CSV is written.
set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
OUT_DIR=fl_dermamnist/results/mu_sensitivity_flower
PARTITION=balanced_paired_7_clients
LOCAL_EPOCHS=20

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
PROX_MUS=(0.001 0.01 0.1 1.0)
EXTRA_ARGS="--log-update-norms"

submit() {
  local algo="$1" mu="$2" seed="$3"
  sbatch \
    --job-name="muS_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" "$EXTRA_ARGS"
  sleep 1
}

echo "Submitting FedAvg (mu=0) baseline arm, 10 seeds..."
for s in "${SEEDS[@]}"; do
  submit fedavg 0.0 "$s"
done

echo "Submitting FedProx arms, 4 mu values x 10 seeds..."
for mu in "${PROX_MUS[@]}"; do
  for s in "${SEEDS[@]}"; do
    submit fedprox "$mu" "$s"
  done
done

echo ""
echo "Submitted mu-sensitivity sweep: 10 FedAvg(mu=0) + 4 mu x 10 seeds = 50 jobs."
echo "Output directory: $OUT_DIR"
