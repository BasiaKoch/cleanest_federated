#!/bin/bash
set -euo pipefail

# Dissertation-focused HPC submission script.
#
# Scientific question:
#   "How does class heterogeneity across simulated DermaMNIST clients affect
#    FedAvg, and does FedProx plus selected class-imbalance mitigation improve
#    global, per-class, and per-client performance?"
#
# This script submits a curated subset of experiments rather than the full
# matrix in submit_all_core.sh / submit_mitigation.sh / submit_ablation.sh.

LOG_DIR="/home/bk489/federated/federated-thesis/experiments/dermamnist/logs"
mkdir -p "$LOG_DIR"

CORE_CONFIGS=(
  configs/fedavg_iid.yaml
  configs/fedavg_dir05.yaml
  configs/fedprox_dir05.yaml
  configs/fedavg_dir01.yaml
  configs/fedprox_dir01.yaml
  configs/fedavg_pathological.yaml
  configs/fedprox_pathological.yaml
)

MITIGATION_CONFIGS=(
  configs/mitigation/fedprox_dir05_weighted.yaml
  configs/mitigation/fedprox_dir05_focal.yaml
  configs/mitigation/fedavg_dir05_weighted.yaml
)

SEEDS=(42 123 456)

for config in "${CORE_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="$(basename "$config" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$config" "$seed"
    sleep 1
  done
done

for config in "${MITIGATION_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="mit_$(basename "$config" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$config" "$seed"
    sleep 1
  done
done

bash scripts/submit_ablation.sh mu 42

echo ""
echo "Submitted dissertation-focused experiment set:"
echo "- 7 core configs × 3 seeds = 21 jobs"
echo "- 3 mitigation configs × 3 seeds = 9 jobs"
echo "- 1 μ-ablation job"
echo "Total: 31 submissions plus μ sweep"
