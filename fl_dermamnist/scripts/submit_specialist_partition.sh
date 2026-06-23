#!/bin/bash
# Submit the specialist-partition defensive sweep to SLURM.
#
# This sweep is the pre-registered counterfactual to the headline
# balanced_paired_7_clients result. It tests whether the +0.027 macro-F1
# FedProx advantage is specifically a paired-co-training phenomenon or a
# more general label-skew benefit. Same 10 paired seeds, same federation
# (K=7, C=1.0, R=150, E=20), same model, same optimiser, same μ=0.01,
# same Flower runtime — only the partition function changes from
# balanced_paired_7_clients (each minority class held by TWO clients) to
# specialist_7_clients (each minority class held by ONE client). The
# specialist partition is engineered to match balanced_paired's per-
# client n_i EXACTLY, so quantity skew is held constant and the contrast
# isolates the pairing structure.
#
# Pre-registered prediction (timestamp 2026-05-21, committed to
# 09_overleaf_ready.tex \subsection{Specialist-partition pre-registration}
# BEFORE these jobs are submitted):
#     Δ_specialist > 0  AND  |Δ_specialist| < |Δ_paired|
# Honest interpretation of all four possible outcomes is in
# results/thesis_ready/writing/specialist_partition_scenarios.tex.
#
# Sweep size: 10 seeds × {FedAvg, FedProx} = 20 jobs ~25 GPU-h on A100.
# Runtime: Flower 1.x simulation (slurm_template_flower.sh).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
PARTITION=specialist_7_clients
OUT_DIR=fl_dermamnist/results/specialist_partition

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_${algo}_specialist_mu${mu}_E${LOCAL_EPOCHS}_s${seed}" \
    "$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed partition=$PARTITION"
    FAILED+=("$algo $mu $seed")
  fi
  sleep 3
}

for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s"
  submit fedprox $MU  "$s"
done

echo ""
echo "Submitted specialist-partition sweep:"
echo "  - $PARTITION × 10 seeds × 2 algos = 20 jobs → $OUT_DIR"
echo "  Total: ~25 GPU-hours on A100."
echo ""
echo "When complete, run:"
echo "  PYTHONPATH=. python fl_dermamnist/results/thesis_ready/scripts/analyse_specialist_partition.py"
echo "Monitor with:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
