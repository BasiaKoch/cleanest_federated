#!/bin/bash
set -euo pipefail

# Specialist-client experiment.
#
# Each of 7 clients holds ALL training samples of exactly one DermaMNIST class.
# Setup follows Zhao et al. 2018 ("Federated Learning with Non-IID Data") and
# the extreme pathological case in Li et al. 2020 (FedProx).
#
# Hypothesis: with such extreme drift (each client only ever sees one class),
# FedAvg should fail badly. FedProx with a sufficiently large μ should pull
# clients back toward the global model and recover useful classification.
#
# Comparison set:
#   - FedAvg baseline                  (expected to collapse to majority class)
#   - FedProx μ=0.01                   (paper default)
#   - FedProx μ=0.1                    (stronger proximal pull)
#   - FedProx μ=1.0                    (very strong, may under-train)
#
# 4 configs × 3 seeds = 12 jobs, ~6-10 min each.

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
CONFIGS=(
  configs/fedavg_specialist.yaml
  configs/fedprox_specialist_mu001.yaml
  configs/fedprox_specialist_mu01.yaml
  configs/fedprox_specialist_mu10.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="$(basename "$cfg" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$cfg" "$seed"
    sleep 1
  done
done

echo ""
echo "Submitted specialist-client experiment:"
echo "  - 4 configs × 3 seeds = 12 jobs"
echo "  - 7 clients (one per DermaMNIST class), natural class sizes preserved"
echo "  - 100 rounds, local_epochs=5, full participation"
echo "  - μ sweep: FedAvg (μ=0), FedProx μ ∈ {0.01, 0.1, 1.0}"
echo ""
echo "After all jobs finish, generate the plot:"
echo "  python experiments/plot_specialist.py"
