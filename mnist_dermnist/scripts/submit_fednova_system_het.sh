#!/bin/bash
# Submit the FedNova baseline sweep for the system-heterogeneity section.
#
# FedNova (Wang et al., 2020, NeurIPS) is the canonical comparator for
# the heterogeneous-local-step regime: it normalises client updates by
# the momentum-aware effective coupling constant
#   a_i = (1 - m^{tau_i}) / (1 - m)
# so that clients running fewer local steps don't contribute disproportionately
# small updates after the size-weighted aggregation. The thesis FedAvg-vs-FedProx
# system-het sweep covers the proximal-anchor mechanism; this script adds
# FedNova as the direct competitor.
#
# Scope of this submission
# ------------------------
# Single condition C2 (random stragglers, Li et al. 2020 setup) ×
# 10 paired seeds = 10 jobs. Headline algorithms FedAvg and FedProx
# at C2 are already covered by submit_system_het.sh; this adds the
# third arm of the three-way comparison.
#
# Total compute: ~10 GPU-h on A100.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
PARTITION=balanced_paired_7_clients

FAILED=()
submit() {
  local seed="$1" out="$2" sh_mode="$3" extra="$4"
  if ! sbatch \
    --job-name="mn_fednova_${sh_mode}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_fednova.sh" \
    "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$sh_mode" "$extra"; then
    echo "  FAILED to submit: seed=$seed sh=$sh_mode"
    FAILED+=("$seed $sh_mode")
  fi
  sleep 3
}

# --- C2: random_stragglers (Li-style; the canonical FedNova test) ---
OUT=mnist_dermnist/results/system_het_random_fednova
mkdir -p "$REPO_ROOT/$OUT"
for s in "${SEEDS[@]}"; do
  submit "$s" "$OUT" random_stragglers "--straggler-fraction 0.5"
done

echo ""
echo "Submitted FedNova sweep:"
echo "  - random_stragglers × 10 seeds = 10 jobs → $OUT"
echo "  Total: 10 jobs ~10 GPU-hours."
echo ""
echo "When complete, combine with the existing FedAvg / FedProx C2 results:"
echo "  PYTHONPATH=. python mnist_dermnist/results/thesis_ready_system_het/scripts/analyse_three_way.py"
echo "  (script to be written when the 10-job sweep lands)"
echo ""
echo "Monitor with:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
