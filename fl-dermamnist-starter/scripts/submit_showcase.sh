#!/bin/bash
set -euo pipefail

# FedProx best-case comparison.
#
# Hypothesis: FedProx's proximal term should show measurable advantage when
# client drift is maximised. We stack three drift amplifiers:
#   - Dirichlet α=0.1     (severe label skew)
#   - local epochs = 20   (clients overfit locally — drift compounds)
#   - fraction_fit = 0.5  (only 5/10 clients per round — high update variance)
#
# Submits 6 jobs (2 configs × 3 seeds), ~10–12 min each at A100.

LOG_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs
mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
CONFIGS=(
  configs/fedavg_showcase.yaml
  configs/fedprox_showcase.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="$(basename "$cfg" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$cfg" "$seed"
    sleep 1
  done
done

echo ""
echo "Submitted FedProx best-case showcase:"
echo "  - 2 configs × 3 seeds = 6 jobs"
echo "  - Dir(0.1), E=20, fraction_fit=0.5, 100 rounds"
echo ""
echo "After all jobs finish (~30-45 min wall-clock), generate the plot:"
echo "  python experiments/plot_showcase.py"
