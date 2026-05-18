#!/bin/bash
# Submit the system-heterogeneity sweep to SLURM.
#
# Two conditions × 10 seeds × 2 algorithms = 40 jobs at ~1 GPU-hour each.
# Both conditions use the same balanced_paired_7_clients partition as the
# headline sweep, so the system-heterogeneity effect can be isolated by
# comparing within-pair Δ to the headline Δ (= +0.027, p = 0.020).
#
#   C1 "fixed_stragglers"  — clients 5 and 6 always do E=5 (the rest do E=20).
#                            Simpler design; client identities of stragglers
#                            are deterministic. Inspired by Marija (2025) §3.8.4.
#   C2 "random_stragglers" — each round, 50% of clients are randomly designated
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
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_system_het.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$sh_mode" "$extra_args"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed sh=$sh_mode"
    FAILED+=("$algo $mu $seed $sh_mode")
  fi
  sleep 3
}

# --- C1: fixed_stragglers (clients 5,6 always E=5) ---
C1_OUT=mnist_dermnist/results/system_het_fixed
mkdir -p "$REPO_ROOT/$C1_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$C1_OUT" fixed_stragglers "--straggler-epochs $STRAGGLER_EPOCHS --fixed-straggler-ids 5,6"
  submit fedprox $MU  "$s" "$C1_OUT" fixed_stragglers "--straggler-epochs $STRAGGLER_EPOCHS --fixed-straggler-ids 5,6"
done

# --- C2: random_stragglers (Li-style, 50% stragglers per round) ---
C2_OUT=mnist_dermnist/results/system_het_random
mkdir -p "$REPO_ROOT/$C2_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$C2_OUT" random_stragglers "--straggler-fraction 0.5"
  submit fedprox $MU  "$s" "$C2_OUT" random_stragglers "--straggler-fraction 0.5"
done

echo ""
echo "Submitted system-heterogeneity sweep:"
echo "  - fixed_stragglers (C5,C6 at E=$STRAGGLER_EPOCHS) × 10 seeds × 2 algos = 20 jobs → $C1_OUT"
echo "  - random_stragglers (50% per round) × 10 seeds × 2 algos = 20 jobs → $C2_OUT"
echo "  Total: 40 jobs ~40 GPU-hours."
echo "Monitor with: squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
