#!/bin/bash
set -euo pipefail

# Thesis headline experiment: paired FedAvg vs FedProx (μ=0.1) at E=20.
# 10 paired seeds — same seed → same initialization → same client partition
# → same Flower sampling schedule for both algorithms.

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

# 10 paired seeds — spec requires ≥10 for Wilcoxon power
SEEDS=(42 123 456 789 999 2024 31337 8675309 271828 161803)

CONFIGS=(
  configs/thesis/fedavg.yaml
  configs/thesis/fedprox_mu01.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="thesis_$(basename "$cfg" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$cfg" "$seed"
    sleep 1
  done
done

echo ""
echo "Submitted thesis headline experiment:"
echo "  - 2 configs (FedAvg vs FedProx μ=0.1) × 10 seeds = 20 jobs"
echo "  - E=20, partial participation (5/10 clients/round), 150 rounds"
echo "  - Each job ~25-35 min on A100 (longer because E=20 + bigger CNN)"
echo "  - Total wall-clock estimate: 1.5-3 hours depending on parallelism"
echo ""
echo "When done, run:"
echo "  python analysis/thesis_stats.py"
echo "  python analysis/thesis_plots.py"
