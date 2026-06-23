#!/bin/bash
# Permanent heterogeneous-compute sweep — Wang et al. (2020) FedNova-style
# system heterogeneity on the engineered partition.
#
# 10 seeds × 2 algorithms × 2 partitions = 40 jobs at ~1 GPU-hour each on
# the Cambridge HPC ampere partition. To run only the engineered arm
# (the load-bearing system-het experiment), comment out the IID block at
# the bottom.
#
# Schedule: each client draws a permanent local-epoch budget E_i once at
# experiment start, from the discrete set {2, 5, 10, 15, 20}. The budget
# is held fixed for all 150 communication rounds. The pair (seed, algo)
# share the same schedule because build_epoch_schedule is keyed on seed,
# so paired within-seed differences reflect only the proximal-term gating.
#
# Why this design (vs the existing fixed_stragglers / random_stragglers):
#  - Mirrors Wang et al. (2020) FedNova §5: realistic deployment in which
#    different institutions have permanently different compute capability.
#  - The (drift × γ-inexact) regime where Li et al. (2020) Theorem 4
#    predicts the proximal term to provide the largest convergence
#    benefit: bounded local dissimilarity (engineered partition) AND
#    heterogeneous inexact local work (permanent E_i).
#  - Removes the per-round randomness of random_stragglers, which can
#    masquerade as ordinary across-seed variance.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20            # E_max; ceiling for the permanent draws
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
SH_MODE=permanent_stragglers
# FedNova-style 5-level compute set. Keeps E_max = 20 reachable.
PERMANENT_CHOICES="2,5,10,15,20"

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3" out="$4" partition="$5" extra_args="$6"
  if ! sbatch \
    --job-name="mn_${algo}_perm_mu${mu}_s${seed}_$(basename "$partition")" \
    "$REPO_ROOT/infra/slurm/slurm_template_system_het.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$out" "$partition" "$SH_MODE" "$extra_args"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed partition=$partition"
    FAILED+=("$algo $mu $seed $partition")
  fi
  sleep 3
}

n_eng=0
n_iid=0

# === Engineered partition (PRIMARY) ====================================
# This is the load-bearing run. Bounded dissimilarity is present
# (engineered partition has drift), and permanent E_i introduces the
# γ-inexact regime — the (B > 1) × (γ < 1) case in Li et al. Theorem 4.
PARTITION_ENG=balanced_paired_7_clients
ENG_OUT=fl_dermamnist/results/system_het_permanent_engineered
mkdir -p "$REPO_ROOT/$ENG_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$ENG_OUT" "$PARTITION_ENG" \
    "--permanent-epoch-choices $PERMANENT_CHOICES --log-update-norms"
  submit fedprox $MU  "$s" "$ENG_OUT" "$PARTITION_ENG" \
    "--permanent-epoch-choices $PERMANENT_CHOICES --log-update-norms"
  n_eng=$((n_eng + 2))
done

# === IID partition (ISOLATION) =========================================
# Same protocol on IID. Pair with the engineered arm above to attribute
# any FedProx benefit either to drift control (engineered − IID) or to
# pure heterogeneous-compute handling (IID alone). Comment out if compute
# is tight; the engineered arm is the primary run.
PARTITION_IID=iid_7_clients
IID_OUT=fl_dermamnist/results/system_het_permanent_iid
mkdir -p "$REPO_ROOT/$IID_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s" "$IID_OUT" "$PARTITION_IID" \
    "--permanent-epoch-choices $PERMANENT_CHOICES --log-update-norms"
  submit fedprox $MU  "$s" "$IID_OUT" "$PARTITION_IID" \
    "--permanent-epoch-choices $PERMANENT_CHOICES --log-update-norms"
  n_iid=$((n_iid + 2))
done

echo ""
echo "Submitted permanent heterogeneous-compute sweep:"
total=0
if [ "$n_eng" -gt 0 ]; then
  echo "  - Engineered (PRIMARY): $n_eng jobs → $ENG_OUT"
  total=$((total + n_eng))
fi
if [ "$n_iid" -gt 0 ]; then
  echo "  - IID (isolation):      $n_iid jobs → $IID_OUT"
  total=$((total + n_iid))
fi
echo "  Total: $total jobs (~$total GPU-hours on A100)"
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Analyse afterward by extending analyse_statistical_heterogeneity.py"
echo "with the two new output dirs, or by computing within-pair Δ directly"
echo "from test_at_best_*.json files in the standard manner used elsewhere"
echo "in the dissertation."

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
