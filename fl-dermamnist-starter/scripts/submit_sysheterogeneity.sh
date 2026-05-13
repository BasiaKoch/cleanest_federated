#!/bin/bash
set -euo pipefail

# System heterogeneity comparison — the canonical FedProx scenario.
# Each round, each client is assigned a random local-epoch count from
# {1, 2, 5, 10, 20}. Some clients are stragglers, others over-trainers.
# This is the regime the FedProx paper (Fig. 4) shows largest gap.
#
# Submits 6 jobs (2 configs × 3 seeds).
# Per-job runtime is variable (some rounds have heavy clients):
# expect 10–20 min per job.

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
CONFIGS=(
  configs/fedavg_sysheterogeneity.yaml
  configs/fedprox_sysheterogeneity.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="$(basename "$cfg" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$cfg" "$seed"
    sleep 1
  done
done

echo ""
echo "Submitted system-heterogeneity comparison:"
echo "  - 2 configs × 3 seeds = 6 jobs"
echo "  - Dir(α=0.3), local_epochs ∈ {1, 2, 5, 10, 20} sampled per-client per-round"
echo ""
echo "After all jobs finish, generate the plot:"
echo "  python experiments/plot_sysheterogeneity.py"
