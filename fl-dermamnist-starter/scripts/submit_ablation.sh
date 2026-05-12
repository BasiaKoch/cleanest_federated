#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/submit_ablation.sh <ablation_name> [seed]"
  echo "  ablation_name: local_epochs | mu | alpha | clients | participation"
  exit 1
fi

seed="${2:-42}"
sbatch --job-name="ablation_${1}_s${seed}" scripts/slurm_ablation.sh "$1" "$seed"
