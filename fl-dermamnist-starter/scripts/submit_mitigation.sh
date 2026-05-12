#!/bin/bash
set -euo pipefail

mkdir -p logs
for config in configs/mitigation/*.yaml; do
  for seed in 42 123 456; do
    sbatch --job-name="mit_$(basename "$config" .yaml)_s${seed}" \
      scripts/slurm_template.sh "$config" "$seed"
    sleep 1
  done
done

echo "Submitted mitigation experiments"
