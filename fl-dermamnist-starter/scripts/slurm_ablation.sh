#!/bin/bash
#SBATCH -J fl_ablation
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter/logs/%x_%j.err

set -euo pipefail

REPO_DIR=/home/bk489/federated_clean/cleanest_federated/fl-dermamnist-starter
cd "$REPO_DIR"
source "$REPO_DIR/.venv/bin/activate"

export RAY_TMPDIR="$HOME/ray_tmp"
mkdir -p "$RAY_TMPDIR" "$REPO_DIR/logs"

python experiments/run_ablations.py --ablation "$1" --seed "$2" --device cuda
