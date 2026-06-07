#!/bin/bash
#SBATCH -J mn_derm_fn
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --exclude=gpu-q-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/%x_%j.err
#
# NOTE: gpu-q-31 is excluded because its /var/spool was full during the 2026-06-05
# run, causing heredoc preflight to fail with "No space left on device" inside
# slurm_script:line 49 before Python even started. Drop the exclusion once the
# node is confirmed clean.

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
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

# Preflight check — use `python -c` instead of a heredoc so we don't depend on
# a writable /tmp (gpu-q-31 hit "No space left on device" on the heredoc temp
# file on 2026-06-05 and failed all jobs at this preflight step).
python -c 'import flwr, torch; print(f"Preflight OK: flwr={flwr.__version__}, torch={torch.__version__}")'

# Retry loop. We do NOT use `if cmd; then ...; fi` here because under
# `set -e`, when the `if` test fails and the `else` branch is empty, the `if`
# compound is considered "tested negatively" and Bash drops `$?` to the
# compound's exit (often 0) by the time we read it after the `fi`. That hid
# the real Python exit code on the 2026-06-05 run, producing the misleading
# "Attempt N failed with exit code 0" message. Capture `rc` immediately after
# the command instead.
RETRY_MAX=3
SUCCESS=0
rc=0
for attempt in $(seq 1 $RETRY_MAX); do
    echo "=== Attempt $attempt/$RETRY_MAX at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

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

    if [ "$attempt" -lt "$RETRY_MAX" ]; then
        echo "Sleeping 90s to let GPU recover before retry..."
        sleep 90
    fi
done

if [ "$SUCCESS" -eq 0 ]; then
    echo "All $RETRY_MAX attempts failed; last Python exit code: $rc. seed=$SEED sh=$SH_MODE"
    exit "${rc:-1}"
fi
