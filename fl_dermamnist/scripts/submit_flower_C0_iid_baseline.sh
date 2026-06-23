#!/bin/bash
# IID-PARTITION C0 BASELINE for system-heterogeneity H2 inference.
#
# Why this exists:
# The engineered-partition C0 baseline at results/flower_C0_baseline/
# already has a modest FedProx tilt (Δ ≈ +0.007 macro-F1) because
# balanced_paired_7_clients is non-IID by construction. Under that
# baseline, the H2 contrast Δ_c - Δ_C0 mixes the system-heterogeneity
# amplification with the residual statistical-heterogeneity effect.
#
# An IID-partition C0 baseline should produce Δ ≈ 0 by the
# bounded-dissimilarity argument (Li et al. 2020 §4.1): when local
# objectives F_i are i.i.d. samples of the same global distribution,
# the proximal anchor has no client drift to constrain. Running C0,
# C1, C2 on the IID partition then isolates the pure system-het
# contribution from the statistical-het contribution.
#
# This is the C0 IID baseline; pair with submit_system_het_iid.sh
# for the C1 and C2 IID arms.
#
# Hyperparameters: identical to submit_flower_C0_baseline.sh except
# for the partition.
#   • Partition: iid_7_clients   (was: balanced_paired_7_clients)
#   • Algorithms: FedAvg (μ=0.0), FedProx (μ=0.01), FedNova (m=0.9)
#   • Seeds: 42 123 456 789 999 2024 31337 8675309 161803 271828
#   • R = 150 rounds, E = 20 local epochs, batch = 32, lr = 0.01
#   • Full participation (C = 1.0), uniform local-epoch schedule
#   • --log-update-norms enabled
#
# Compute: 30 jobs (10 seeds × 3 algorithms), ~30 GPU-hours on A100.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
PARTITION=iid_7_clients
OUT_DIR=fl_dermamnist/results/flower_C0_iid_baseline

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

EXTRA_ARGS="--log-update-norms"

FAILED=()
submit_flower() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_C0iid_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" "$EXTRA_ARGS"; then
    echo "  FAILED: $algo mu=$mu seed=$seed"
    FAILED+=("$algo $mu $seed")
  fi
  sleep 3
}

submit_fednova() {
  local seed="$1"
  if ! sbatch \
    --job-name="mn_C0iid_fednova_s${seed}" \
    "$REPO_ROOT/fl_dermamnist/scripts/slurm_template_fednova.sh" \
    "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" uniform "$EXTRA_ARGS"; then
    echo "  FAILED: fednova seed=$seed"
    FAILED+=("fednova $seed")
  fi
  sleep 3
}

for s in "${SEEDS[@]}"; do
  submit_flower fedavg  0.0  "$s"
  submit_flower fedprox $MU  "$s"
  submit_fednova "$s"
done

echo ""
echo "Submitted IID C0 baseline:"
echo "  - 10 seeds × {FedAvg, FedProx, FedNova} = 30 jobs → $OUT_DIR"
echo "  Total: ~30 GPU-hours."
echo ""
echo "Once complete, these are the IID-partition C0 numbers."
echo "Pair with submit_system_het_iid.sh outputs (C1, C2 on IID partition)"
echo "to test the H2 contrast Δ_c - Δ_C0 on an unbiased baseline where"
echo "Δ_C0 is expected to be ≈ 0 (no inter-client drift to constrain)."

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
