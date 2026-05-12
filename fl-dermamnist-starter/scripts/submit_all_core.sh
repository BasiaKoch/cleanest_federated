#!/bin/bash
set -euo pipefail

mkdir -p /home/bk489/federated/federated-thesis/experiments/dermamnist/logs

for config in configs/fedavg_*.yaml configs/fedprox_*.yaml; do
  for seed in 42 123 456; do
    sbatch --job-name="$(basename "$config" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$config" "$seed"
    sleep 1
  done
done

echo "Submitted all core experiments"
