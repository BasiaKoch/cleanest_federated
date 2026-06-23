#!/bin/bash
#SBATCH -J mn_derm_pt
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/%x_%j.err

# Pure-PyTorch SLURM template — wraps `fl_dermamnist.experiments.run_one`,
# the sequential server-loop reference runner that produces the
# pure-PyTorch headline data (results/headline/). Mirrors the CLI of
# slurm_template_flower.sh so the same args work.
#
# Args:
#   $1 = algorithm  (fedavg | fedprox)
#   $2 = mu         (0.0 for fedavg, e.g. 0.01 for fedprox)
#   $3 = seed
#   $4 = local_epochs
#   $5 = out_dir
#   $6 = partition
#   $7 = extra args (optional, e.g. "--log-update-norms")

set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

ALGO="${1:?algorithm required}"
MU="${2:?mu required}"
SEED="${3:?seed required}"
LOCAL_EPOCHS="${4:?local_epochs required}"
OUT_DIR="${5:?out_dir required}"
PARTITION="${6:?partition required}"
EXTRA_ARGS="${7:-}"

mkdir -p "$OUT_DIR" fl_dermamnist/logs

if [ ! -f "$REPO_ROOT/dermamnist_64.npz" ]; then
    echo "ERROR: dataset not found at $REPO_ROOT/dermamnist_64.npz" >&2
    exit 2
fi

# Retry loop mirroring slurm_template_flower.sh for parity
RETRY_MAX=3
SUCCESS=0
for attempt in $(seq 1 $RETRY_MAX); do
    echo "=== Attempt $attempt/$RETRY_MAX at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    if PYTHONPATH=. python -m fl_dermamnist.experiments.run_one \
            --algorithm "$ALGO" \
            --mu "$MU" \
            --seed "$SEED" \
            --local-epochs "$LOCAL_EPOCHS" \
            --num-rounds 150 \
            --lr 0.01 \
            --batch-size 10 \
            --momentum 0.0 \
            --partition "$PARTITION" \
            --device cuda \
            --npz-path "$REPO_ROOT/dermamnist_64.npz" \
            --out-dir "$OUT_DIR" \
            $EXTRA_ARGS; then
        SUCCESS=1
        echo "Job complete on attempt $attempt (pure-PyTorch): algo=$ALGO mu=$MU seed=$SEED E=$LOCAL_EPOCHS partition=$PARTITION"
        break
    fi
    rc=$?
    echo "Attempt $attempt failed with exit code $rc"
    if [ $attempt -lt $RETRY_MAX ]; then
        echo "Sleeping 90s to let GPU recover before retry..."
        sleep 90
    fi
done
if [ "$SUCCESS" -eq 0 ]; then
    echo "All $RETRY_MAX attempts failed; giving up. algo=$ALGO mu=$MU seed=$SEED"
    exit 1
fi
