#!/bin/bash
#SBATCH -J mn_derm_fn
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.err

# SLURM template for one FedNova run, mirroring slurm_template_flower.sh.
# Args:
#   $1 = seed
#   $2 = local_epochs
#   $3 = out_dir
#   $4 = partition
#   $5 = system_het_mode  (uniform | fixed_stragglers | random_stragglers)
#   $6 = extra args (e.g. "--straggler-fraction 0.5")

set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

SEED="${1:?seed required}"
LOCAL_EPOCHS="${2:?local_epochs required}"
OUT_DIR="${3:?out_dir required}"
PARTITION="${4:?partition required}"
SH_MODE="${5:?system_het_mode required}"
EXTRA_ARGS="${6:-}"

mkdir -p "$OUT_DIR" mnist_dermnist/logs

if [ ! -f "$REPO_ROOT/dermamnist_64.npz" ]; then
    echo "ERROR: dataset not found at $REPO_ROOT/dermamnist_64.npz" >&2
    exit 2
fi

python - <<'PY'
import flwr
import torch
print(f"Preflight OK: flwr={flwr.__version__}, torch={torch.__version__}")
PY

PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_fednova_flower \
    --seed "$SEED" \
    --local-epochs "$LOCAL_EPOCHS" \
    --num-rounds 150 \
    --lr 0.01 \
    --batch-size 32 \
    --partition "$PARTITION" \
    --device cuda \
    --npz-path "$REPO_ROOT/dermamnist_64.npz" \
    --out-dir "$OUT_DIR" \
    --system-het-mode "$SH_MODE" \
    $EXTRA_ARGS

echo "Job complete (FedNova): seed=$SEED E=$LOCAL_EPOCHS partition=$PARTITION sh=$SH_MODE"
