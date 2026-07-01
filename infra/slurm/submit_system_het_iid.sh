#!/bin/bash
# IID-PARTITION system-heterogeneity sweep - C1 and C2 on iid_7_clients.
#
# Two conditions × 10 seeds × 2 algorithms = 40 jobs at ~1 GPU-hour each.
#
# Why this exists:
# The engineered-partition system-het sweep (submit_system_het.sh)
# tests the H2 contrast Δ_c - Δ_C0 against a non-IID baseline that
# already has a modest FedProx tilt. To cleanly isolate the
# system-heterogeneity contribution from the statistical-heterogeneity
# contribution, this script re-runs C1 and C2 on the IID partition.
# Pair with the IID C0 baseline from submit_flower_C0_iid_baseline.sh.
#
#   C1 "fixed_stragglers"  - clients 5 and 6 always do E=5 (others do E=20).
#                            Under IID, client identities are exchangeable,
#                            so the choice of (5, 6) is arbitrary; kept
#                            consistent with the engineered-partition C1
#                            for direct cross-partition comparison.
#   C2 "random_stragglers" - each round, 50% of clients are randomly
#                            designated stragglers with E_i ~ Uniform[1, 19];
#                            others do E=20. Li et al. (2020) §5.2.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
PARTITION=iid_7_clients
STRAGGLER_EPOCHS=5

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3" out="$4" sh_mode="$5" extra_args="$6"
  if ! sbatch \
    --job-name="mn_iid_${algo}_${sh_mode}_mu${mu}_s${seed}" \
    "$REPO_ROOT/infra/slurm/slurm_template_system_het.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$sh_mode" "$extra_args"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed sh=$sh_mode"
    FAILED+=("$algo $mu $seed $sh_mode")
  fi
  sleep 3
}

n_c1_submitted=0
n_c2_submitted=0

# === C1: Fixed stragglers on IID partition ===========================
# Under IID partitioning, client identities are exchangeable (each
# client gets a uniform random sample of the global distribution), so
# the choice of permanent-straggler ids carries no structural meaning.
# We keep (5, 6) to match the engineered-partition C1, enabling a
# direct cross-partition comparison.
C1_OUT=fl_dermamnist/results/system_het_iid_fixed
mkdir -p "$REPO_ROOT/$C1_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$C1_OUT" fixed_stragglers "--straggler-epochs $STRAGGLER_EPOCHS --fixed-straggler-ids 5,6 --log-update-norms"
  submit fedprox $MU  "$s" "$C1_OUT" fixed_stragglers "--straggler-epochs $STRAGGLER_EPOCHS --fixed-straggler-ids 5,6 --log-update-norms"
  n_c1_submitted=$((n_c1_submitted + 2))
done

# === C2: Random stragglers on IID partition (PRIMARY for IID H2) =====
# Li et al. (2020) §5.2 random-straggler protocol on the IID partition.
# Under IID + random stragglers, FedAvg and FedProx are predicted to be
# closer to matched than under the engineered partition: the IID
# baseline Δ ≈ 0 means the H2 contrast Δ_C2_iid - Δ_C0_iid is the pure
# system-heterogeneity amplification, uncontaminated by residual
# statistical-heterogeneity.
C2_OUT=fl_dermamnist/results/system_het_iid_random
mkdir -p "$REPO_ROOT/$C2_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$C2_OUT" random_stragglers "--straggler-fraction 0.5 --log-update-norms"
  submit fedprox $MU  "$s" "$C2_OUT" random_stragglers "--straggler-fraction 0.5 --log-update-norms"
  n_c2_submitted=$((n_c2_submitted + 2))
done

echo ""
echo "Submitted IID system-heterogeneity sweep:"
total=0
if [ "$n_c1_submitted" -gt 0 ]; then
  echo "  - C1_iid fixed_stragglers (C5,C6 at E=$STRAGGLER_EPOCHS): $n_c1_submitted jobs → $C1_OUT"
  total=$((total + n_c1_submitted))
fi
if [ "$n_c2_submitted" -gt 0 ]; then
  echo "  - C2_iid random_stragglers (50% per round): $n_c2_submitted jobs → $C2_OUT"
  total=$((total + n_c2_submitted))
fi
echo "  Total: $total jobs (~$total GPU-hours on A100)"
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Once complete, analyse alongside submit_flower_C0_iid_baseline.sh"
echo "output (results/flower_C0_iid_baseline/) using a small extension"
echo "to analyse_system_heterogeneity.py that ingests the *_iid_* dirs."

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
