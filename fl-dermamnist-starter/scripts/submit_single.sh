#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch scripts/slurm_template.sh <config> <seed>"
  echo "Example: sbatch scripts/slurm_template.sh configs/fedavg_dir05.yaml 42"
  exit 1
fi

sbatch scripts/slurm_template.sh "$1" "$2"
