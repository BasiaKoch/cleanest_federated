#!/bin/bash
# μ sweep (HIGHLY RECOMMENDED before claiming the headline result).
# Per Li et al. 2020: μ ∈ {0.001, 0.01, 0.1, 0.5, 1.0} should be swept on validation.
# 3 seeds per μ + 3 FedAvg(μ=0.0) sanity baselines = 18 jobs at E=20.
# The μ=0.0 FedAvg row is a built-in validation: it must match the existing
# headline FedAvg numbers exactly (within the cross-runtime noise floor),
# since FedProx(μ=0) ≡ FedAvg by the gated proximal-term branch.
set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
OUT_DIR=fl_dermamnist/results/mu_sweep
PARTITION=balanced_paired_7_clients   # explicit; template default is the same
mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

SEEDS=(42 123 456)
LOCAL_EPOCHS=20
MUS=(0.001 0.01 0.1 0.5 1.0)

submit() {
  local algo="$1" mu="$2" seed="$3"
  sbatch \
    --job-name="mu_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/infra/slurm/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION"
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
echo "Submitted μ sweep: 3 FedAvg(μ=0.0) + 5 μ × 3 seeds = 18 jobs."
