#!/bin/bash
#SBATCH -J mn_derm_sp
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/%x_%j.err

# Single-process SLURM template - generic wrapper for non-federated
# PyTorch jobs (local-only baselines, fine-tuning, centralised training,
# etc.). Does NOT use Flower or Ray, so it side-steps the GCS-startup
# issues seen with the federated runs. Reuses the same retry-on-CUDA-busy
# logic as slurm_template_flower.sh.
#
# Args:
#   $1 = python module path (e.g. "fl_dermamnist.experiments.run_local_only")
#   $2 = extra args, passed verbatim to the python invocation
#
# Example (manual):
#   sbatch --job-name=mn_local_only_c1_s42 \
#     infra/slurm/slurm_template_single_process.sh \
#     fl_dermamnist.experiments.run_local_only \
#     "--seed 42 --partition two_client_90_10_rare_stress --client-id 1 \
#      --num-epochs 3000 --device cuda \
#      --out-dir fl_dermamnist/results/small_hospital_local_only"

set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

MODULE="${1:?python module path required}"
EXTRA_ARGS="${2:-}"

mkdir -p fl_dermamnist/logs

if [ ! -f "$REPO_ROOT/dermamnist_64.npz" ]; then
    echo "ERROR: dataset not found at $REPO_ROOT/dermamnist_64.npz" >&2
    exit 2
fi

python - <<'PY'
import torch
print(f"Preflight OK: torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
PY

# Retry CUDA-busy transients (same logic as slurm_template_flower.sh).
RETRY_MAX=3
SUCCESS=0
for attempt in $(seq 1 $RETRY_MAX); do
    echo "=== Attempt $attempt/$RETRY_MAX at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    if PYTHONPATH=. python -m "$MODULE" $EXTRA_ARGS; then
        SUCCESS=1
        echo "Job complete on attempt $attempt (single-process): module=$MODULE"
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
    echo "All $RETRY_MAX attempts failed; giving up. module=$MODULE"
    exit 1
fi
