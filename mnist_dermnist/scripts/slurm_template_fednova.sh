#!/bin/bash
#SBATCH -J mn_derm_fn
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --exclude=gpu-q-31,gpu-q-32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.err
#
# NOTE: gpu-q-31 was excluded after its /var/spool filled up on 2026-06-05;
# gpu-q-32 was added on 2026-06-07 after a smoke test reproduced the same
# Ray "Failed to start GCS" failure mode there (job 30215930). Both nodes
# appear to share a degraded /tmp or /dev/shm state. Drop entries once the
# nodes are confirmed clean by HPC ops.

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

# Per-job Ray session dir — see slurm_template_flower.sh for rationale.
# This must match slurm_template_flower.sh exactly because that path has
# worked for every prior FedNova/Flower experiment on this HPC; deviating
# from the known-good path created new failures during the 2026-06-05 probe
# attempt.
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
# Postmortem: copy the Ray session logs to a durable location before
# the trap removes RAY_TMPDIR. Diagnoses GCS startup failures (which
# leave gcs_server.out empty in the .err message but write to other
# files in the session dir).
ray_log_save() {
    local dest="$REPO_ROOT/mnist_dermnist/logs/ray_session_${SLURM_JOB_ID}"
    if [ -d "$RAY_TMPDIR" ]; then
        mkdir -p "$dest"
        cp -r "$RAY_TMPDIR"/* "$dest/" 2>/dev/null || true
    fi
    rm -rf "$RAY_TMPDIR"
}
trap ray_log_save EXIT

# Preflight import check. Use `python -c` (not a heredoc) so it does NOT
# depend on the compute node being able to create a heredoc temp file —
# the gpu-q-31/gpu-q-32 failures were node-local /tmp problems, and a
# heredoc preflight could itself fail to stage on a degraded node.
python -c 'import flwr, torch; print(f"Preflight OK: flwr={flwr.__version__}, torch={torch.__version__}")'

# Retry loop — see slurm_template_flower.sh for rationale.
RETRY_MAX=3
SUCCESS=0
for attempt in $(seq 1 $RETRY_MAX); do
    echo "=== Attempt $attempt/$RETRY_MAX at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    # Capture the REAL python exit code. The previous `if python ...; then`
    # pattern is broken under `set -e`: when the command fails the `if`
    # compound returns 0 (no else branch ran), so a trailing `rc=$?` always
    # read 0 and every failure was logged as "exit code 0". Disable -e only
    # around the run, grab $? immediately, then re-enable.
    set +e
    PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_fednova_flower \
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
            $EXTRA_ARGS
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        SUCCESS=1
        echo "Job complete on attempt $attempt (FedNova): seed=$SEED E=$LOCAL_EPOCHS partition=$PARTITION sh=$SH_MODE"
        break
    fi
    echo "Attempt $attempt failed with exit code $rc"
    if [ $attempt -lt $RETRY_MAX ]; then
        echo "Sleeping 90s to let GPU recover before retry..."
        sleep 90
    fi
done
if [ "$SUCCESS" -eq 0 ]; then
    echo "All $RETRY_MAX attempts failed; giving up. seed=$SEED sh=$SH_MODE"
    exit 1
fi
