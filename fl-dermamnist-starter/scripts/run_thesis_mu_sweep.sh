#!/bin/bash
set -euo pipefail

# μ sweep at E=20 — 3 seeds per μ value.
# Used to validate μ=0.1 was a defensible choice (and to make Plot F).

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
MU_CONFIGS=(
  configs/thesis/fedavg.yaml             # μ=0 baseline
  configs/thesis/fedprox_mu_0001.yaml    # μ=0.001
  configs/thesis/fedprox_mu_001.yaml     # μ=0.01
  configs/thesis/fedprox_mu01.yaml       # μ=0.1
  configs/thesis/fedprox_mu_05.yaml      # μ=0.5
  configs/thesis/fedprox_mu_1.yaml       # μ=1.0
  configs/thesis/fedprox_mu_2.yaml       # μ=2.0
)

for cfg in "${MU_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="thesis_$(basename "$cfg" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$cfg" "$seed"
    sleep 1
  done
done

echo ""
echo "Submitted μ sweep:"
echo "  - 7 configs × 3 seeds = 21 jobs"
