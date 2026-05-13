#!/bin/bash
set -euo pipefail
# Patch 4 — heterogeneity sweep at fixed E=10, 5 seeds.

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456 789 999)
LEVELS=(low medium high extreme)

for level in "${LEVELS[@]}"; do
  for algo in fedavg fedprox; do
    cfg="configs/thesis/${algo}_het_${level}.yaml"
    for seed in "${SEEDS[@]}"; do
      sbatch --job-name="thesis_${algo}_het_${level}_s${seed}" \
        scripts/slurm_template.sh "$cfg" "$seed"
      sleep 1
    done
  done
done

echo ""
echo "Submitted heterogeneity sweep:"
echo "  - 4 levels × 2 algos × 5 seeds = 40 jobs"
echo "  - specialist_dominance ∈ {0.2, 0.4, 0.6, 0.9}"
