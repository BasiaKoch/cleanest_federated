#!/bin/bash
# μ-sweep across the heterogeneity ladder — STAGE A pilot (one seed).
#
# Tests the hypothesis from the discussion of the ladder result that
# FedProx's μ = 0.01 is miscalibrated for these DermaMNIST partitions.
# A per-partition μ sweep is the experiment Li et al. (2020) actually
# recommend ("we tune μ ∈ {0.001, 0.01, 0.1, 1, 5} … no default μ values
# would work for all settings").
#
# Submits 12 SLURM jobs:
#
#   μ-values  ×  Levels                = 12 jobs
#   ------------------------------------
#   {0.001, 0.01, 0.1, 1.0}  ×  {L0, L2, L4}
#
# Levels chosen to span the JS-divergence range:
#   L0  IID 50/50                     JS = 0.000  (null baseline)
#   L2  Label-skew 50/50              JS = 0.104  (mid heterogeneity)
#   L4  Severe 90/10 rare-stress      JS = 0.385  (where FedProx should help)
#
# Stage A runs are 1 seed only (seed 42); promote winners to 3 seeds
# in Stage B if any (level, μ*) combination is interesting.
#
# Outputs land in their own directory so they don't collide with the
# existing ladder results (where only μ = 0.01 is present):
#
#   fl_dermamnist/results/mu_sweep_ladder/L{0,2,4}_<partition>/
#       test_at_best_fedprox_mu{0.001,0.01,0.1,1.0}_E20_s42.json
#
# Usage
# -----
#   bash fl_dermamnist/scripts/submit_mu_sweep_ladder.sh            # submit
#   DRY_RUN=1 bash fl_dermamnist/scripts/submit_mu_sweep_ladder.sh  # print only
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

SEED=42
LOCAL_EPOCHS=20
NUM_ROUNDS=150

FLOWER_TPL="$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh"

# 3 ladder levels spanning the JS-divergence range.
LEVELS=(0 2 4)
PARTITIONS=(
    two_client_50_50_stratified_iid
    two_client_50_50_label_skew_only
    two_client_90_10_rare_stress
)

# μ values to sweep. Matches Li et al. (2020) recommended grid minus μ = 5
# (which would essentially freeze local updates and is uninformative).
MU_VALUES=(0.001 0.01 0.1 1.0)

DRY_RUN="${DRY_RUN:-0}"

JOBS_SUMMARY=()
FAILED=()

submit_flower() {
    local mu="$1" partition="$2" outdir="$3" jobname="$4"
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FLOWER_TPL"
        "fedprox" "$mu" "$SEED" "$LOCAL_EPOCHS"
        "$outdir" "$partition"
        "--num-rounds $NUM_ROUNDS --log-update-norms")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY-RUN: ${cmd[*]}"
        echo "  -> would submit  $jobname"
        return
    fi
    local jid
    if ! jid=$("${cmd[@]}"); then
        echo "  FAILED: $jobname" >&2
        FAILED+=("$jobname")
        return 1
    fi
    JOBS_SUMMARY+=("$jid  $jobname")
    echo "  $jid  $jobname"
}

echo "============================================================"
echo "FedProx μ-sweep across heterogeneity ladder (seed $SEED)"
echo "  Levels: ${LEVELS[*]}"
echo "  μ:      ${MU_VALUES[*]}"
echo "  Total:  $(( ${#LEVELS[@]} * ${#MU_VALUES[@]} )) jobs"
echo "============================================================"

for i in "${!LEVELS[@]}"; do
    LEVEL=${LEVELS[$i]}
    PARTITION=${PARTITIONS[$i]}
    OUTDIR="fl_dermamnist/results/mu_sweep_ladder/L${LEVEL}_${PARTITION}"
    mkdir -p "$REPO_ROOT/$OUTDIR"

    echo ""
    echo "--- Level $LEVEL: $PARTITION ---"

    for MU in "${MU_VALUES[@]}"; do
        submit_flower "$MU" "$PARTITION" "$OUTDIR" \
            "mn_muswp_L${LEVEL}_mu${MU}_s${SEED}"
    done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete (no jobs submitted)."
    echo "Set DRY_RUN=0 to actually submit."
else
    echo "Submitted ${#JOBS_SUMMARY[@]} jobs (${#FAILED[@]} failed)."
    for entry in "${JOBS_SUMMARY[@]}"; do echo "  $entry"; done
fi
echo "============================================================"
echo "Monitor with: squeue -u \$USER --format='%.10i %.45j %.8T %.20R'"
echo "Outputs in:   fl_dermamnist/results/mu_sweep_ladder/L<level>_<partition>/"
echo ""
echo "Next:  after all 12 jobs finish, on the Mac run"
echo "         python fl_dermamnist/analysis/analyse_mu_sweep_ladder.py"
echo "       It reads from mu_sweep_ladder/, re-uses the existing FedAvg"
echo "       baselines from heterogeneity_ladder/, and emits:"
echo "         - mu_sweep_summary.csv  (long format, level × μ × metrics)"
echo "         - mu_sweep_pivot.csv    (μ × level macro-F1 grid)"
echo "         - F_mu_sweep_ladder.{pdf,png}  (curves + Δ-bars vs FedAvg)"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
