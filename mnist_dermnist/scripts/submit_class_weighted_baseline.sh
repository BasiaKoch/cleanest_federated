#!/bin/bash
# Submit the loss-side imbalance baseline (audit HV2).
#
# Why this exists
# ---------------
# The headline FedProx-vs-FedAvg result is a paired contrast at fixed
# loss (standard cross-entropy). A reasonable reviewer question is
# whether the +0.027 macro-F1 advantage reflects FedProx specifically,
# or just the absence of imbalance-aware loss-side mitigation. To
# answer it, this script runs the same 10 paired seeds with FedAvg +
# class-weighted CE (inverse-frequency weights computed once from the
# global training-set class counts). The expected comparison:
#
#   FedAvg + CE                (headline FedAvg)         macro-F1 ~ 0.481
#   FedAvg + class-weighted CE (this sweep)              macro-F1 ~ ?
#   FedProx + CE               (headline FedProx)        macro-F1 ~ 0.508
#
# Interpretation:
#   - If CW-CE >= FedProx:    the +0.027 is loss-side, not optimisation-side
#   - If CW-CE <  FedAvg + CE: imbalance-aware loss alone is not enough
#   - If CW-CE in between:    FedProx and CW-CE address different problems
#
# Compute: 10 jobs (FedAvg + CW-CE), ~10 GPU-hours on A100.
#
# We deliberately do NOT also run FedProx + CW-CE: the question is
# whether the loss-side fix on its own competes with FedProx, not
# whether combining them helps further.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
PARTITION=balanced_paired_7_clients
OUT_DIR=mnist_dermnist/results/class_weighted_baseline

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/mnist_dermnist/logs"

FAILED=()
submit() {
  local seed="$1"
  if ! sbatch \
    --job-name="mn_cwce_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh" \
    fedavg 0.0 "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" \
    "--loss-type class_weighted_ce"; then
    echo "  FAILED: seed=$seed"
    FAILED+=("$seed")
  fi
  sleep 3
}

for s in "${SEEDS[@]}"; do
  submit "$s"
done

echo ""
echo "Submitted class-weighted CE baseline:"
echo "  - FedAvg + CW-CE × 10 seeds = 10 jobs → $OUT_DIR"
echo "  Total: ~10 GPU-hours."
echo ""
echo "When complete, compare against:"
echo "  mnist_dermnist/results/headline/                (FedAvg + plain CE)"
echo "  mnist_dermnist/results/headline/                (FedProx + plain CE)"
echo ""
echo "Analysis: PYTHONPATH=. python -m mnist_dermnist.analysis.tables \\"
echo "          --results-dir $OUT_DIR"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
