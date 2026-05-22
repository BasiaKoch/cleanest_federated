#!/bin/bash
# Cambridge HPC SLURM submission for the Li et al. 2020 §5.2 asymmetric
# straggler-handling protocol.
#
# Hypothesis being tested
# ------------------------
# The original FedProx paper claims FedProx significantly outperforms
# FedAvg under random stragglers when:
#   - FedAvg DROPS straggler clients' updates (γ-inexact work is discarded)
#   - FedProx INCLUDES straggler updates (proximal anchor provides stability
#     guarantee for γ-inexact contributions)
#
# Our symmetric C2 sweep (system_het_random/) treats both algorithms
# identically and finds only Δ = +0.017 (n.s. at n=10). This experiment
# tests whether the asymmetric protocol — the literature-canonical one —
# produces a clearly significant FedProx advantage on DermaMNIST.
#
# Submitted jobs (20 total)
# -------------------------
#   10 paired seeds * 1 algorithm * --drop-stragglers (FedAvg-drops-stragglers)
#   10 paired seeds * 1 algorithm * NO flag             (FedProx-includes-stragglers)
#
# Output directory: mnist_dermnist/results/system_het_random_asymmetric/
# Filename convention: FedAvg files get '_drop' suffix; FedProx files unchanged
#
# Expected outcome (per literature)
# ---------------------------------
# Predicted Δ_asymmetric = +0.05 to +0.15 macro-F1 with p < 0.05.
# This would be the "FedProx wins clearly" result the existing
# symmetric protocol does not deliver.
#
# Run AFTER pulling the commit that adds:
#   - mnist_dermnist/fl_flower/strategy_straggler_dropping.py
#   - --drop-stragglers flag in run_one_flower.py
#
# Usage on HPC login node:
#   cd /home/bk489/federated_clean/cleanest_federated
#   git pull origin main
#   bash mnist_dermnist/scripts/hpc_asymmetric_stragglers.sh
#   squeue -u $USER
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TFLW="$REPO/mnist_dermnist/scripts/slurm_template_flower.sh"
OUT=mnist_dermnist/results/system_het_random_asymmetric
E=20
MU=0.01
PAIRED=balanced_paired_7_clients

# Paired-seed protocol — same 10 seeds as the headline + system_het sweeps
SEEDS=(42 123 456 789 999 2024 31337 161803 271828 8675309)

cd "$REPO"

# Sanity checks
if ! grep -q "numpy_legacy_seed" mnist_dermnist/fl_flower/client.py; then
  echo "ERROR: HPC checkout missing seed-overflow fix." >&2
  exit 2
fi
if ! grep -q "StragglerDroppingFedAvg" mnist_dermnist/experiments/run_one_flower.py; then
  echo "ERROR: HPC checkout missing --drop-stragglers wiring." >&2
  echo "Pull commit with strategy_straggler_dropping.py first." >&2
  exit 2
fi
if ! grep -q -- "--drop-stragglers" mnist_dermnist/experiments/run_one_flower.py; then
  echo "ERROR: --drop-stragglers flag not in runner." >&2
  exit 2
fi

mkdir -p "$OUT" mnist_dermnist/logs

echo "============================================================"
echo " HPC asymmetric-straggler protocol (Li 2020 §5.2)"
echo "============================================================"
echo " partition: $PAIRED"
echo " system-het mode: random_stragglers (50% of clients per round)"
echo " seeds: ${SEEDS[*]}"
echo " algos: fedavg --drop-stragglers ; fedprox (no flag)"
echo " out_dir: $OUT"
echo " expected: 20 jobs"
echo "============================================================"
echo ""

# Common args for both algorithms
RANDOM_EXTRA_BASE="--straggler-fraction 0.5 --log-update-norms --system-het-mode random_stragglers"

# FedAvg side: drops stragglers (Li 2020 §5.2 FedAvg behavior)
for SEED in "${SEEDS[@]}"; do
    echo "Submitting FedAvg --drop-stragglers seed=$SEED ..."
    EXTRA="$RANDOM_EXTRA_BASE --drop-stragglers"
    sbatch "$TFLW" fedavg 0.0 "$SEED" "$E" "$OUT" "$PAIRED" "$EXTRA"
    sleep 3
done

# FedProx side: includes stragglers (γ-inexact via proximal anchor)
for SEED in "${SEEDS[@]}"; do
    echo "Submitting FedProx (no drop) seed=$SEED ..."
    sbatch "$TFLW" fedprox "$MU" "$SEED" "$E" "$OUT" "$PAIRED" "$RANDOM_EXTRA_BASE"
    sleep 3
done

echo ""
echo "============================================================"
echo " Submitted 20 jobs total (10 FedAvg-drop + 10 FedProx-include)"
echo " Watch with: squeue -u \$USER"
echo " After completion, analyse the asymmetric-protocol delta with:"
echo "   python mnist_dermnist/scripts/check_asymmetric_stragglers.py"
echo "============================================================"
