#!/bin/bash
# Stage-3 expansion (FEDNOVA_RANDOM_TAU_THESIS_PLAN.md §5/§7): top up the
# discriminating arms identified by the Stage-2 mechanism-fork pilot from
# n=5 to the full n=10 thesis seed set.
#
# Per §5: "Expand only the winning/discriminating arms to 10 seeds... Do NOT
# expand every arm; expand the arm(s) that move the needle and the baseline."
#
# Pilot verdict (analysed 2026-06-07): tauclip320 is the standout arm — the
# only intervention that structurally changes the amplification factor
# (a_eff/a_i: ~37x -> ~1.9x; client 6 dominates 0/750 rounds vs ~5% in every
# other arm) and the closest to the pre-registered STRONG bar (final macro-F1
# 0.415 vs the >=0.45 threshold; 1/5 collapse vs the 0/5 threshold) with the
# smallest best-val -> final-round decay (-13% vs -25..37% elsewhere). mom0 /
# servmom / serverlr03 are NOT expanded here: servmom is actively harmful
# (falsifies Hyp C), serverlr03 is the weaker Hyp-A arm, mom0 is a necessity
# check, not a rescue candidate.
#
# NOTE: baseline is expanded too (not just reused from the pre-existing n=10
# `results/system_het_random_fednova/`) because that directory predates the
# §4 instrumentation (no test_at_final / collapse_round / aggregation_client_diag
# — the very columns §9's Wilcoxon tests and §10's F1-F7 figures need), AND
# because a reproducibility check found 3/5 overlapping seeds give different
# numbers across the two commits (2ab422b vs c9ffe4d, both git-dirty). Building
# a single self-consistent instrumented n=10 baseline avoids that confound.
#
# Seeds added (the 5 of the full n=10 set not in the Stage-2 pilot):
#   456 999 2024 161803 789
# These land in the SAME output dirs as the pilot (keyed by seed -> no
# collisions), giving a fully-paired, fully-instrumented n=10 for both arms.
#
# SAFETY: DRY-RUN by default — prints the sbatch commands and DOES NOT submit.
# To actually submit on the cluster:   DRY_RUN=0 bash infra/slurm/submit_fednova_stage3_expand.sh
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
PARTITION=balanced_paired_7_clients
SH_MODE=random_stragglers
SEEDS=(456 999 2024 161803 789)

COMMON="--batch-size 32 --straggler-fraction 0.5 --log-update-norms"

# Arms to expand (index-aligned) — ONLY the winning arm + baseline, per §5.
ARM_NAMES=(baseline tauclip320)
ARM_FLAGS=(
  "--momentum 0.9"
  "--momentum 0.9 --tau-clip-min 320"
)

DRY_RUN="${DRY_RUN:-1}"
ACCOUNT="${ACCOUNT:-FERGUSSON-SL3-GPU}"
FAILED=()
N_PLANNED=0

submit() {
  local seed="$1" out="$2" name="$3" extra="$4"
  local jobname="mn_fn_stage3_${name}_s${seed}"
  local template="$REPO_ROOT/infra/slurm/slurm_template_fednova.sh"
  N_PLANNED=$((N_PLANNED + 1))
  if [ "$DRY_RUN" = "1" ]; then
    printf '  '
    printf '%q ' sbatch --account="$ACCOUNT" --job-name="$jobname" "$template" \
      "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$SH_MODE" "$extra"
    printf '\n'
    return 0
  fi
  mkdir -p "$REPO_ROOT/$out"
  if ! sbatch --account="$ACCOUNT" --job-name="$jobname" "$template" \
       "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$SH_MODE" "$extra"; then
    echo "  FAILED to submit: seed=$seed arm=$name"
    FAILED+=("$seed $name")
  fi
  sleep 3
}

if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY-RUN (no jobs submitted). Set DRY_RUN=0 to submit. ==="
fi

for i in "${!ARM_NAMES[@]}"; do
  name="${ARM_NAMES[$i]}"
  arm_flags="${ARM_FLAGS[$i]}"
  out="fl_dermamnist/results/system_het_random_fednova_${name}"
  extra="$COMMON $arm_flags"
  echo ""
  echo "--- arm: ${name}   out: ${out}   extra: ${extra} ---"
  for s in "${SEEDS[@]}"; do
    submit "$s" "$out" "$name" "$extra"
  done
done

echo ""
echo "==============================================================="
echo "Stage-3 expansion: ${#ARM_NAMES[@]} arms x ${#SEEDS[@]} seeds = ${N_PLANNED} runs"
echo "  seeds:   ${SEEDS[*]}  (tops up the n=5 pilot to the full n=10 set)"
echo "  arms:    ${ARM_NAMES[*]}"
echo "  out:     fl_dermamnist/results/system_het_random_fednova_{${ARM_NAMES[*]// /,}}/  (merges with pilot data)"
echo "  account: ${ACCOUNT}  partition: ampere"
echo "  ~14 min/run on A100  =>  ~2.3 GPU-h total"
if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "DRY-RUN only — nothing submitted. To submit on the cluster:"
  echo "  DRY_RUN=0 bash infra/slurm/submit_fednova_stage3_expand.sh"
fi
echo ""
echo "When runs complete, you'll have a fully-instrumented n=10 for baseline"
echo "and tauclip320 -- re-run the amplification analysis and the §9 stats"
echo "(paired wins/losses, Wilcoxon signed-rank vs baseline, collapse rate)."
echo "Monitor with:  squeue -u \$USER"

if [ "${#FAILED[@]}" -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
