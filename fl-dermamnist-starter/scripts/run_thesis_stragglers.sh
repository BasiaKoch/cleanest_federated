#!/bin/bash
set -euo pipefail
# Patch 5 — stragglers experiment.
# Replicates FedProx paper Figure 1 on DermaMNIST.

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
FRACS=(00 05 09)
ALGOS=(fedavg fedprox0 fedprox)

for frac in "${FRACS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    cfg="configs/thesis/${algo}_str_${frac}.yaml"
    for seed in "${SEEDS[@]}"; do
      sbatch --job-name="thesis_str_${algo}_${frac}_s${seed}" \
        scripts/slurm_template.sh "$cfg" "$seed"
      sleep 1
    done
  done
done

echo ""
echo "Submitted stragglers experiment:"
echo "  - 3 fractions × 3 algos × 3 seeds = 27 jobs"
echo "  - Replicates FedProx paper Figure 1 (3-way comparison)"
echo ""
echo "Three-way decomposition:"
echo "  fedavg_*    : drops stragglers (paper's FedAvg)"
echo "  fedprox0_*  : keeps stragglers, μ=0 (isolates partial-work benefit)"
echo "  fedprox_*   : keeps stragglers, μ=0.1 (isolates proximal-term benefit)"
