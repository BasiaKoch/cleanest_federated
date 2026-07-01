#!/bin/bash
# Stage-2 mechanism-fork PILOT (FEDNOVA_RANDOM_TAU_THESIS_PLAN.md §6).
#
# Submits the 5-seed pilot {42, 123, 8675309, 31337, 271828} across 5 arms on
# the random-straggler / balanced-paired-7-clients protocol, to adjudicate the
# three mechanism hypotheses (A magnitude / B direction / C jitter):
#
#   arm          extra flags                       output dir suffix        tests
#   ----------   -------------------------------   ----------------------   --------------------------
#   baseline     --momentum 0.9                    _baseline                reproduce macro-F1 ~0.237
#   mom0         --momentum 0.0                    _mom0                    is client momentum necessary?
#   tauclip320   --momentum 0.9 --tau-clip-min 320 _tauclip320              Hyp. B (direction domination)
#   servmom      --momentum 0.9 --server-momentum 0.9  _servmom             Hyp. C (round-to-round jitter)
#   serverlr03   --momentum 0.9 --server-lr 0.3    _serverlr03              Hyp. A (effective-LR magnitude)
#
# Decision rules (plan §8): server-lr fixes & tau-clip doesn't => magnitude;
# tau-clip fixes & server-lr doesn't => direction; both => mixed; neither =>
# cumulative random-tau instability. A rescue is STRONG only if final-round
# macro-F1 mean >= 0.45 AND collapse rate 0/N.
#
# NOTE on the protocol override: slurm_template_fednova.sh hardcodes
# `--batch-size 10 --momentum 0.0`, but the baseline collapse runs used
# batch=32, momentum=0.9 (see results/system_het_random_fednova/test_at_best_*.json).
# Argparse takes the LAST occurrence, so the per-arm extras below explicitly set
# --batch-size 32 and --momentum {0.9|0.0} to match the baseline protocol exactly.
#
# tau_i is the local SGD STEP count, not epochs: with balanced_paired_7_clients
# (~1001 samples/client) and batch=32, batches_per_epoch ~ 32, so --tau-clip-min
# 320 ~= 10 epochs (E_max/2) in step units. `10` would be a no-op (min tau ~22).
#
# Compute: 5 arms x 5 seeds = 25 runs x ~14 min on A100 ~= 6 GPU-h.
#
# SAFETY: DRY-RUN by default - prints the sbatch commands and DOES NOT submit.
# To actually submit on the cluster:   DRY_RUN=0 bash infra/slurm/submit_fednova_mechanism_fork_pilot.sh
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
PARTITION=balanced_paired_7_clients
SH_MODE=random_stragglers
SEEDS=(42 123 8675309 31337 271828)

# Common protocol flags for every arm (override template hardcodes + log diag).
COMMON="--batch-size 32 --straggler-fraction 0.5 --log-update-norms"

# Arms (index-aligned). ARM_FLAGS already includes the arm's --momentum value.
ARM_NAMES=(baseline mom0 tauclip320 servmom serverlr03)
ARM_FLAGS=(
  "--momentum 0.9"
  "--momentum 0.0"
  "--momentum 0.9 --tau-clip-min 320"
  "--momentum 0.9 --server-momentum 0.9"
  "--momentum 0.9 --server-lr 0.3"
)

# DRY-RUN by default. Set DRY_RUN=0 to actually sbatch.
DRY_RUN="${DRY_RUN:-1}"
# Slurm account (overridable). Switched to FERGUSSON-SL3-GPU after
# MPHIL-DIS-SL2-GPU ran out of GPU-minutes (AssocGrpGRESMinutes). The CLI
# --account below overrides the template's #SBATCH -A directive; override per
# run with e.g.  ACCOUNT=SOME-OTHER-SL2-GPU DRY_RUN=0 bash <this script>.
ACCOUNT="${ACCOUNT:-FERGUSSON-SL3-GPU}"
FAILED=()
N_PLANNED=0

submit() {
  local seed="$1" out="$2" name="$3" extra="$4"
  local jobname="mn_fn_pilot_${name}_s${seed}"
  local template="$REPO_ROOT/infra/slurm/slurm_template_fednova.sh"
  N_PLANNED=$((N_PLANNED + 1))
  if [ "$DRY_RUN" = "1" ]; then
    # Print a copy-pasteable, shell-escaped sbatch invocation.
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
echo "Mechanism-fork pilot: ${#ARM_NAMES[@]} arms x ${#SEEDS[@]} seeds = ${N_PLANNED} runs"
echo "  seeds:  ${SEEDS[*]}"
echo "  arms:   ${ARM_NAMES[*]}"
echo "  out:    fl_dermamnist/results/system_het_random_fednova_{${ARM_NAMES[*]// /,}}/"
echo "  account: ${ACCOUNT}  partition: ampere"
echo "  ~14 min/run on A100  =>  ~6 GPU-h total"
if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "DRY-RUN only — nothing submitted. To submit on the cluster:"
  echo "  DRY_RUN=0 bash infra/slurm/submit_fednova_mechanism_fork_pilot.sh"
fi
echo ""
echo "When runs complete, analyse with:"
echo "  for d in baseline mom0 tauclip320 servmom serverlr03; do"
echo "    PYTHONPATH=. python -m fl_dermamnist.analysis.amplification \\"
echo "      --dir fl_dermamnist/results/system_het_random_fednova_\$d \\"
echo "      --out fl_dermamnist/results/system_het_random_fednova_\$d/analysis/amplification"
echo "  done"
echo "Monitor with:  squeue -u \$USER"

if [ "${#FAILED[@]}" -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
