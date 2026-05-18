#!/bin/bash
#SBATCH -J mn_derm_fl
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.err

# Flower-runtime SLURM template (drop-in replacement for slurm_template.sh).
# Same CLI as the pure-PyTorch template; routes the run through
# `mnist_dermnist.experiments.run_one_flower` which uses
# `flwr.simulation.start_simulation` under the hood.
#
# Args (identical to slurm_template.sh):
#   $1 = algorithm  (fedavg | fedprox)
#   $2 = mu         (0.0 for fedavg, e.g. 0.01 for fedprox)
#   $3 = seed
#   $4 = local_epochs
#   $5 = out_dir
#   $6 = partition

set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

ALGO="${1:?algorithm required}"
MU="${2:?mu required}"
SEED="${3:?seed required}"
LOCAL_EPOCHS="${4:?local_epochs required}"
OUT_DIR="${5:-mnist_dermnist/results/headline}"
PARTITION="${6:-balanced_paired_7_clients}"

mkdir -p "$OUT_DIR" mnist_dermnist/logs

PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \
    --algorithm "$ALGO" \
    --mu "$MU" \
    --seed "$SEED" \
    --local-epochs "$LOCAL_EPOCHS" \
    --num-rounds 150 \
    --lr 0.01 \
    --batch-size 32 \
    --partition "$PARTITION" \
    --device cuda \
    --npz-path "$REPO_ROOT/dermamnist_64.npz" \
    --out-dir "$OUT_DIR"

echo "Job complete (Flower runtime): algo=$ALGO mu=$MU seed=$SEED E=$LOCAL_EPOCHS partition=$PARTITION"
