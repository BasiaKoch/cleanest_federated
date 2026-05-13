#!/bin/bash
# E sweep — 5 E values × 2 algos × 3 seeds = 30 jobs at the recommended μ=0.1.
set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
OUT_DIR=mnist_dermnist/results/e_sweep
mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/mnist_dermnist/logs"

SEEDS=(42 123 456)
ES=(1 5 10 20 40)

submit() {
  local algo="$1" mu="$2" seed="$3" E="$4"
  sbatch \
    --job-name="E${E}_${algo}_mu${mu}_s${seed}" \
    --time=10:00:00 \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template.sh" \
    "$algo" "$mu" "$seed" "$E" "$OUT_DIR"
  sleep 1
}

for E in "${ES[@]}"; do
  for s in "${SEEDS[@]}"; do
    submit fedavg  0.0 "$s" "$E"
    submit fedprox 0.1 "$s" "$E"
  done
done

echo ""
echo "Submitted E sweep: 5 E × 2 algos × 3 seeds = 30 jobs."
