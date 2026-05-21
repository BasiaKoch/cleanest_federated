#!/bin/bash
#SBATCH -J mn_centralised
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.err

set -euo pipefail
REPO=/home/bk489/federated_clean/cleanest_federated
cd "$REPO"
source /home/bk489/federated_clean/.venv/bin/activate

SEED="${1:-42}"
EPOCHS="${2:-50}"
OUT="${3:-mnist_dermnist/results/centralised}"

mkdir -p "$OUT" mnist_dermnist/logs

if [ ! -f "$REPO/dermamnist_64.npz" ]; then
    echo "ERROR: dataset not found at $REPO/dermamnist_64.npz" >&2
    exit 2
fi

# Centralised baseline is pure-PyTorch by definition (not federated);
# only torch is required (no flwr import).
python - <<'PY'
import torch
print(f"Preflight OK: torch={torch.__version__}")
PY

PYTHONPATH=. python -m mnist_dermnist.experiments.run_centralised \
    --seed "$SEED" \
    --num-epochs "$EPOCHS" \
    --device cuda \
    --npz-path "$REPO/dermamnist_64.npz" \
    --out-dir "$OUT"
echo "Centralised job complete: seed=$SEED epochs=$EPOCHS"
