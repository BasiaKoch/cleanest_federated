#!/bin/bash
set -euo pipefail

# E sweep — FedAvg vs FedProx at E ∈ {1, 5, 10, 40} with 3 seeds each.
# E=20 is the headline (separate script with 10 seeds).
# E=1 will run fast; E=40 will be the slowest (~50 min/job).

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
CONFIGS=(
  configs/thesis/fedavg_E1.yaml      configs/thesis/fedprox_E1.yaml
  configs/thesis/fedavg_E5.yaml      configs/thesis/fedprox_E5.yaml
  configs/thesis/fedavg_E10.yaml     configs/thesis/fedprox_E10.yaml
  configs/thesis/fedavg_E40.yaml     configs/thesis/fedprox_E40.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="thesis_$(basename "$cfg" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$cfg" "$seed"
    sleep 1
  done
done

echo ""
echo "Submitted E sweep:"
echo "  - 8 configs × 3 seeds = 24 jobs"
echo "  - E ∈ {1, 5, 10, 40} for both FedAvg and FedProx (μ=0.1)"
