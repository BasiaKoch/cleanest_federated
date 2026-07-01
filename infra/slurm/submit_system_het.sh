#!/bin/bash
# Submit the system-heterogeneity sweep to SLURM.
#
# Two conditions × 10 seeds × 2 algorithms = 40 jobs at ~1 GPU-hour each.
# Both conditions use the same balanced_paired_7_clients partition as the
# headline sweep, so the system-heterogeneity effect can be isolated by
# comparing within-pair Δ to the headline Δ (= +0.027, p = 0.020).
#
#   C1 "fixed_stragglers"  - clients 5 and 6 always do E=5 (the rest do E=20).
#                            Simpler design; client identities of stragglers
#                            are deterministic.
#   C2 "random_stragglers" - each round, 50% of clients are randomly designated
#                            stragglers with E_i ~ Uniform[1, 19]; others do E=20.
#                            Follows Li et al. (2020) §5.2 exactly.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
PARTITION=balanced_paired_7_clients
STRAGGLER_EPOCHS=5

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3" out="$4" sh_mode="$5" extra_args="$6"
  if ! sbatch \
    --job-name="mn_${algo}_${sh_mode}_mu${mu}_s${seed}" \
    "$REPO_ROOT/infra/slurm/slurm_template_system_het.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$sh_mode" "$extra_args"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed sh=$sh_mode"
    FAILED+=("$algo $mu $seed $sh_mode")
  fi
  sleep 3
}

# C1 and C2 are independent blocks. To run C2 only (e.g. when compute is
# tight), comment out the entire C1 block below. The trailing summary
# print at the bottom of this script adapts to which blocks executed.

n_c1_submitted=0
n_c2_submitted=0

# === C1: Fixed stragglers (DESCRIPTIVE ONLY - confounded) ============
# NOTE: C1 is confounded - clients 5,6 are structurally tied to the
# melanoma/nevi class mechanism in balanced_paired_7_clients. C5 holds
# the second melanoma-pair (389 melanoma + 49 vascular + 670 nevi);
# C6 is the nevi-only generalist (673 nevi). Making C5 a permanent
# straggler partially neutralises the per-class mechanism that drives
# the headline result, so C1 will likely UNDER-report FedProx's
# advantage relative to a non-confounded straggler choice. Treat C1
# numbers as descriptive cross-check only; C2 is the primary
# system-heterogeneity condition for inference.
#
# To skip C1 entirely (recommended if compute is tight or for the
# "C2-only" plan), comment out lines 50-58 (the entire C1 block).
C1_OUT=fl_dermamnist/results/system_het_fixed
mkdir -p "$REPO_ROOT/$C1_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$C1_OUT" fixed_stragglers "--straggler-epochs $STRAGGLER_EPOCHS --fixed-straggler-ids 5,6 --log-update-norms"
  submit fedprox $MU  "$s" "$C1_OUT" fixed_stragglers "--straggler-epochs $STRAGGLER_EPOCHS --fixed-straggler-ids 5,6 --log-update-norms"
  n_c1_submitted=$((n_c1_submitted + 2))
done

# === C2: Random stragglers (PRIMARY condition) =======================
# Li et al. (2020, MLSys) §5.2 random-straggler protocol:
#   • Each round: 50% of the 7 clients (= 4 clients) are randomly
#     designated stragglers (sampled without replacement).
#   • Stragglers perform E_i ~ Uniform{1, 2, ..., 19} local epochs.
#   • Non-stragglers perform E = 20 local epochs.
#   • Straggler identity ROTATES per round (seed-deterministic).
# Selection is independent of class composition - no structural
# coupling with the per-class mechanism that drives the headline.
# C2 is the primary system-heterogeneity condition; H2 = Δ_C2 - Δ_C0
# is the inferentially primary test (see analyse_system_het.py).
C2_OUT=fl_dermamnist/results/system_het_random
mkdir -p "$REPO_ROOT/$C2_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$C2_OUT" random_stragglers "--straggler-fraction 0.5 --log-update-norms"
  submit fedprox $MU  "$s" "$C2_OUT" random_stragglers "--straggler-fraction 0.5 --log-update-norms"
  n_c2_submitted=$((n_c2_submitted + 2))
done

echo ""
echo "Submitted system-heterogeneity sweep:"
total=0
if [ "$n_c1_submitted" -gt 0 ]; then
  echo "  - C1 fixed_stragglers (C5,C6 at E=$STRAGGLER_EPOCHS, DESCRIPTIVE): $n_c1_submitted jobs → $C1_OUT"
  total=$((total + n_c1_submitted))
else
  echo "  - C1: skipped"
fi
if [ "$n_c2_submitted" -gt 0 ]; then
  echo "  - C2 random_stragglers (50% per round, PRIMARY): $n_c2_submitted jobs → $C2_OUT"
  total=$((total + n_c2_submitted))
else
  echo "  - C2: skipped"
fi
echo "  Total: $total jobs (~$total GPU-hours on A100)"
echo "Monitor with: squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
