#!/bin/bash
#SBATCH -J fl_ablation
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/experiments/dermamnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/experiments/dermamnist/logs/%x_%j.err

set -euo pipefail

cd /home/bk489/federated/federated-thesis/fl-dermamnist
source /home/bk489/federated/federated-thesis/fl-dermamnist/.venv/bin/activate

python experiments/run_ablations.py --ablation "$1" --seed "$2" --device cuda
