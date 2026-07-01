#!/bin/bash
# Cambridge HPC SLURM equivalent of runpod_addendum_fednova_c2.sh.
#
# Submits 3 FedNova C2 (random_stragglers) jobs to fill
# `fl_dermamnist/results/system_het_random_fednova/` to 10 paired seeds:
#   - fednova s123     (was MISSING)
#   - fednova s789     (was MISSING)
#   - fednova s8675309 (was the numpy-overflow bug; re-runs with seed fix)
#
# Prerequisite: the HPC checkout must contain the seeding-fix
# (fl_dermamnist/core/seeding.py, used by client_fednova.py). The script
# refuses to submit if the fix is absent.
#
# Idempotent: deletes the known-bad s8675309 JSON before submitting.
#
# Usage on the HPC login node:
#   cd /home/bk489/federated_clean/cleanest_federated
#   git pull origin main
#   bash infra/slurm/hpc_addendum_fednova_c2.sh
#   squeue -u $USER   # to watch
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TFNV="$REPO/infra/slurm/slurm_template_fednova.sh"
OUT="$REPO/fl_dermamnist/results/system_het_random_fednova"
E=20
PAIRED=balanced_paired_7_clients
MODE=random_stragglers
EXTRA="--straggler-fraction 0.5 --log-update-norms"

cd "$REPO"

# Sanity check: seed fix must be present
if ! grep -q "numpy_legacy_seed" fl_dermamnist/runtimes/flower/client_fednova.py; then
  echo "ERROR: HPC checkout does not contain the seed-overflow fix." >&2
  echo "Pull latest main before submitting these jobs." >&2
  exit 2
fi

mkdir -p "$OUT"

# Block 0: delete the bad s8675309 JSON (and companion files) so the
# SLURM job actually re-runs it rather than being skipped by the runner's
# implicit overwrite logic. The runner ALWAYS writes a fresh JSON, but
# this guarantees the laptop-side audit will see the new macro_f1.
echo "=== Block 0: delete known-bad s8675309 FedNova JSON ==="
BAD="$OUT/test_at_best_fednova_mu0.0_E20_sh-random_stragglers_s8675309.json"
if [ -f "$BAD" ]; then
    macro=$(python -c "import json; print(json.load(open('$BAD'))['macro_f1'])" 2>/dev/null || echo "-1")
    is_bad=$(python -c "print(1 if float($macro) < 0.20 else 0)" 2>/dev/null || echo "0")
    if [ "$is_bad" = "1" ]; then
        echo "  DELETE  $BAD  (macro_f1=$macro)"
        stem="fednova_mu0.0_E20_sh-random_stragglers_s8675309"
        rm -f "$BAD" \
              "$OUT/history_${stem}.csv" \
              "$OUT/test_predictions_${stem}.npz" \
              "$OUT/client_update_norms_${stem}.csv"
    else
        echo "  KEEP    $BAD  (macro_f1=$macro — already real training)"
    fi
else
    echo "  GONE    $BAD  (already absent)"
fi
echo ""

# Block 1: submit 3 FedNova C2 jobs via sbatch
echo "=== Block 1: sbatch 3 FedNova C2 jobs ==="
for SEED in 123 789 8675309; do
    echo "Submitting FedNova C2 seed=$SEED ..."
    sbatch "$TFNV" "$SEED" "$E" "$OUT" "$PAIRED" "$MODE" "$EXTRA"
    sleep 3   # spread allocations across the scheduler
done

echo ""
echo "=== Submitted 3 FedNova C2 jobs ==="
echo "Watch with: squeue -u \$USER"
echo "After completion, audit with:"
echo "  python fl_dermamnist/analysis/check_system_het.py"
