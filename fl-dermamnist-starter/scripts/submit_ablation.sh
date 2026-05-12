#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch --time=08:00:00 scripts/slurm_template.sh <ablation_config> <seed>"
  exit 1
fi

seed="${2:-42}"
sbatch --time=08:00:00 scripts/slurm_template.sh "$1" "$seed"
