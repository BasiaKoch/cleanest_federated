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
# Same CLI as the pure-PyTorch template (plus an optional 7th slot for
# additional runner flags); routes through
# `mnist_dermnist.experiments.run_one_flower` which uses
# `flwr.simulation.start_simulation` under the hood.
#
# Args:
#   $1 = algorithm  (fedavg | fedprox)
#   $2 = mu         (0.0 for fedavg, e.g. 0.01 for fedprox)
#   $3 = seed
#   $4 = local_epochs
#   $5 = out_dir
#   $6 = partition
#   $7 = extra args, passed verbatim to run_one_flower.py
#        (e.g. "--log-update-norms" or "--loss-type class_weighted_ce")

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
OUT_DIR="${5:-mnist_dermnist/results/headline}"
PARTITION="${6:-balanced_paired_7_clients}"
EXTRA_ARGS="${7:-}"

mkdir -p "$OUT_DIR" mnist_dermnist/logs

if [ ! -f "$REPO_ROOT/dermamnist_64.npz" ]; then
    echo "ERROR: dataset not found at $REPO_ROOT/dermamnist_64.npz" >&2
    exit 2
fi

# Per-job Ray session directory. Flower's start_simulation() calls
# ray.init() which by default writes to /tmp/ray/session_<ts>_<pid>/. On
# the shared ampere compute nodes this collides when multiple Flower
# jobs land on the same node, and the GCS server fails to bind / write
# logs, producing "RuntimeError: Failed to start GCS" within ~3 minutes.
# Isolating Ray's tmpdir per-SLURM-job removes that contention entirely.
# Ray reads RAY_TMPDIR from the environment at ray.init() time, so
# exporting it here propagates through to Flower without code changes.
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

python - <<'PY'
import flwr
import torch
print(f"Preflight OK: flwr={flwr.__version__}, torch={torch.__version__}")
PY

# Retry loop around the python invocation. CUDA initialisation can fail
# transiently when the node has just released a GPU from a previous job
# ("CUDA-capable device(s) is/are busy or unavailable"). Crashes inside
# the first ~60 seconds are almost always this. We retry up to RETRY_MAX
# times within the same SLURM allocation, sleeping between attempts to
# let the GPU recover. SUCCESS exits 0; exhausting retries exits 1.
RETRY_MAX=3
SUCCESS=0
for attempt in $(seq 1 $RETRY_MAX); do
    echo "=== Attempt $attempt/$RETRY_MAX at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    if PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \
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
        echo "Job complete on attempt $attempt (Flower runtime): algo=$ALGO mu=$MU seed=$SEED E=$LOCAL_EPOCHS partition=$PARTITION extra='$EXTRA_ARGS'"
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
    echo "All $RETRY_MAX attempts failed; giving up. algo=$ALGO seed=$SEED"
    exit 1
fi
