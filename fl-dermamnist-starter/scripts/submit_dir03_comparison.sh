#!/bin/bash
set -euo pipefail

# FedAvg vs FedProx at Dirichlet α=0.3 — the missing intermediate-heterogeneity
# regime between α=0.5 and α=0.1, where FedProx might show a clearer advantage.
#
# Settings (from configs/base.yaml + dir03 override):
#   Partition:   Dirichlet, α = 0.3
#   Clients:     10
#   Local epochs: 5
#   Rounds:      100
#   FedProx μ:   0.01
#
# Submits 6 jobs total (2 configs × 3 seeds), ~5–6 min each.

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
CONFIGS=(
  configs/fedavg_dir03.yaml
  configs/fedprox_dir03.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="$(basename "$cfg" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$cfg" "$seed"
    sleep 1
  done
done

echo ""
echo "Submitted FedAvg-vs-FedProx @ Dir(0.3) comparison:"
echo "  - 2 configs × 3 seeds = 6 jobs"
echo ""
echo "After all jobs finish (~30 min wall-clock), generate the plot:"
echo "  python experiments/plot_dir03_comparison.py"
