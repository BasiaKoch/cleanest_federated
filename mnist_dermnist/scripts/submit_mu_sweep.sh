#!/bin/bash
# μ sweep (HIGHLY RECOMMENDED before claiming the headline result).
# Per Li et al. 2020: μ ∈ {0.001, 0.01, 0.1, 1.0} should be swept on validation.
# 3 seeds per μ + 3 FedAvg baselines = 15 jobs at E=20.
set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
OUT_DIR=mnist_dermnist/results/mu_sweep
mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/mnist_dermnist/logs"

SEEDS=(42 123 456)
LOCAL_EPOCHS=20
MUS=(0.001 0.01 0.1 1.0)

submit() {
  local algo="$1" mu="$2" seed="$3"
  sbatch \
    --job-name="mu_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR"
  sleep 1
}

for s in "${SEEDS[@]}"; do
  submit fedavg 0.0 "$s"
done
for mu in "${MUS[@]}"; do
  for s in "${SEEDS[@]}"; do
    submit fedprox "$mu" "$s"
  done
done

echo ""
echo "Submitted μ sweep: 3 FedAvg + 4 μ × 3 seeds = 15 jobs."
