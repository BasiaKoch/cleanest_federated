#!/bin/bash
# Small-hospital case-study baselines — local-only + fine-tuning.
#
# Six new jobs that complete §Case Study: A Small Site with Unique
# Rare Classes (Section sec:small-hospital). The federated FedAvg and
# FedProx runs on the 2-client 90/10 partition already exist
# (results/two_client_90_10_rare_stress/, seed 42). This script adds:
#
#   1. Local-only Client 0  (run_local_only.py)
#   2. Local-only Client 1  (run_local_only.py)
#   3. FedAvg + Client 0 fine-tuning  (run_finetune.py)
#   4. FedAvg + Client 1 fine-tuning  (run_finetune.py)
#   5. FedProx + Client 0 fine-tuning (run_finetune.py)
#   6. FedProx + Client 1 fine-tuning (run_finetune.py)
#
# This implements the standard FL baselining protocol used by Sheller
# et al. (2018, 2020), Roth et al. (2020), and Pati et al. (2022) for
# the "value of federation" comparison, plus the local-adaptation
# protocol of Yu et al. (2022) "Salvaging Federated Learning by Local
# Adaptation" for the personalised-FL comparison.
#
# Prerequisite
# ------------
# The fine-tuning runs (#3-6) require the federated checkpoints to be
# saved as .pt files. Re-run the existing seed-42 federated jobs once
# more with the new --save-best-checkpoint flag to produce them:
#
#   bash -c '
#     for ALGO_MU in "fedavg 0.0" "fedprox 0.01"; do
#       set -- $ALGO_MU
#       PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \
#         --algorithm "$1" --mu "$2" --seed 42 \
#         --local-epochs 20 --num-rounds 150 \
#         --partition two_client_90_10_rare_stress \
#         --save-best-checkpoint \
#         --device cuda \
#         --out-dir mnist_dermnist/results/two_client_90_10_rare_stress
#     done'
#
# Once the .pt files exist, this script runs everything else.
#
# No SLURM / no Ray / no Flower: every job below is a single-process
# PyTorch training. Runs locally with --device cpu (slow) or on a
# single A100 with --device cuda (fast).
#
# Compute: ~3 GPU-hours total on A100 (two long local-only runs at
# ~1 h each; four short fine-tuning runs at ~10-15 min each).
set -uo pipefail

# Resolve REPO_ROOT robustly whether the script is invoked from the
# repo root, HPC home, or anywhere else.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEED=42
PARTITION=two_client_90_10_rare_stress
FED_DIR=mnist_dermnist/results/two_client_90_10_rare_stress
LOCAL_OUT=mnist_dermnist/results/small_hospital_local_only
FT_OUT=mnist_dermnist/results/small_hospital_finetune
DEVICE=${DEVICE:-cuda}

mkdir -p "$LOCAL_OUT" "$FT_OUT"

run() {
  echo "============================================================"
  echo "[$(date +%H:%M:%S)] $*"
  echo "============================================================"
  PYTHONPATH=. "$@" || { echo "FAILED: $*" >&2; return 1; }
}

# === Local-only baselines (configs 1, 2) =============================
# Compute budget matched to the federation: 150 rounds × 20 local epochs
# = 3000 epochs of cumulative local SGD. Equal compute keeps the
# federation-vs-local comparison about data availability, not training
# time.
run python -m mnist_dermnist.experiments.run_local_only \
  --seed $SEED --partition $PARTITION --client-id 0 \
  --num-epochs 3000 --eval-every 20 \
  --device $DEVICE --out-dir $LOCAL_OUT

run python -m mnist_dermnist.experiments.run_local_only \
  --seed $SEED --partition $PARTITION --client-id 1 \
  --num-epochs 3000 --eval-every 20 \
  --device $DEVICE --out-dir $LOCAL_OUT

# === Fine-tuning baselines (configs 3-6) =============================
# Yu et al. (2022) protocol: 5 epochs of local SGD at lr=0.001 starting
# from the federated best-val checkpoint. The personalisation gap is the
# difference between post-FT and pre-FT macro-F1 (computed and stored
# automatically by run_finetune.py).
CKPT_FA="$FED_DIR/best_state_fedavg_mu0.0_E20_s${SEED}.pt"
CKPT_FP="$FED_DIR/best_state_fedprox_mu0.01_E20_s${SEED}.pt"

for ckpt in "$CKPT_FA" "$CKPT_FP"; do
  if [ ! -f "$ckpt" ]; then
    echo "MISSING CHECKPOINT: $ckpt" >&2
    echo "Re-run the federated seed-42 jobs with --save-best-checkpoint first." >&2
    echo "(See header comment of this script for the exact command.)" >&2
    exit 1
  fi
done

for CID in 0 1; do
  for CKPT in "$CKPT_FA" "$CKPT_FP"; do
    run python -m mnist_dermnist.experiments.run_finetune \
      --checkpoint "$CKPT" \
      --seed $SEED --partition $PARTITION --client-id $CID \
      --num-epochs 5 --lr 0.001 \
      --device $DEVICE --out-dir $FT_OUT
  done
done

echo ""
echo "============================================================"
echo "All small-hospital baselines complete."
echo "Local-only outputs   : $LOCAL_OUT"
echo "Fine-tuning outputs  : $FT_OUT"
echo "============================================================"
