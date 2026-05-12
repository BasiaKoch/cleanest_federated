#!/bin/bash
#SBATCH -J fl_derm
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs/%x_%j.err

set -euo pipefail

REPO_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter
cd "$REPO_DIR"
source "$REPO_DIR/.venv/bin/activate"

# Ray temp dir on home (avoid /tmp quota on shared nodes)
export RAY_TMPDIR="$HOME/ray_tmp"
mkdir -p "$RAY_TMPDIR" "$REPO_DIR/logs"

python experiments/run_experiment.py --config "$1" --seed "$2" --device cuda

echo "Job completed: config=$1 seed=$2"
