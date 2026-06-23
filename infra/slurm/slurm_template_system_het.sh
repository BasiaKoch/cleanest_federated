#!/bin/bash
#SBATCH -J mn_derm_sh
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/%x_%j.err

# System-heterogeneity variant of slurm_template.sh.
# Args:
#   $1 = algorithm  (fedavg | fedprox)
#   $2 = mu         (0.0 for fedavg, e.g. 0.01 for fedprox)
#   $3 = seed
#   $4 = local_epochs
#   $5 = out_dir
#   $6 = partition
#   $7 = system_het_mode  (fixed_stragglers | random_stragglers | permanent_stragglers)
#   $8 = extra args passed verbatim (e.g. "--straggler-epochs 5 --fixed-straggler-ids 5,6"
#                                         or "--permanent-epoch-choices 2,5,10,15,20")

set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

# CRITICAL Ray fix (2026-06-07): clear the module-injected LD_LIBRARY_PATH
# (Intel oneAPI MPI / libfabric / UCX). Those libs break Ray's GCS startup
# ("Failed to start GCS", empty gcs_server.out); ray.init() succeeds only with
# LD_LIBRARY_PATH unset (torch CUDA cu128 wheel is unaffected). See
# ray_gcs_probe.sh. Single-node Flower/Ray needs none of those HPC libs.
unset LD_LIBRARY_PATH

ALGO="${1:?algorithm required}"
MU="${2:?mu required}"
SEED="${3:?seed required}"
LOCAL_EPOCHS="${4:?local_epochs required}"
OUT_DIR="${5:?out_dir required}"
PARTITION="${6:?partition required}"
SH_MODE="${7:?system_het_mode required}"
EXTRA_ARGS="${8:-}"

mkdir -p "$OUT_DIR" fl_dermamnist/logs

if [ ! -f "$REPO_ROOT/dermamnist_64.npz" ]; then
    echo "ERROR: dataset not found at $REPO_ROOT/dermamnist_64.npz" >&2
    exit 2
fi

# Per-job Ray session dir — see slurm_template_flower.sh for rationale.
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

python - <<'PY'
import flwr
import torch
print(f"Preflight OK: flwr={flwr.__version__}, torch={torch.__version__}")
PY

# Retry loop — see slurm_template_flower.sh for rationale.
RETRY_MAX=3
SUCCESS=0
for attempt in $(seq 1 $RETRY_MAX); do
    echo "=== Attempt $attempt/$RETRY_MAX at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    if PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
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
            --system-het-mode "$SH_MODE" \
            $EXTRA_ARGS; then
        SUCCESS=1
        echo "Job complete on attempt $attempt (Flower runtime): algo=$ALGO mu=$MU seed=$SEED E=$LOCAL_EPOCHS partition=$PARTITION sh=$SH_MODE"
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
    echo "All $RETRY_MAX attempts failed; giving up. algo=$ALGO seed=$SEED sh=$SH_MODE"
    exit 1
fi
