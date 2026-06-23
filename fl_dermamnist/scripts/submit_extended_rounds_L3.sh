#!/bin/bash
# Extended-rounds 3-seed L3 — convergence-truncation fix.
#
# The single-seed L3 result has FedProx's best-val round = 150 = the
# final round of training. That means FedProx hadn't plateaued when we
# stopped it. Reporting "FedProx loses at L3" off this run is unfair —
# it's an under-training artefact, not an algorithmic one.
#
# Fix: re-run L3 only, both methods, 3 seeds, with --num-rounds 250.
# The runner's --num-rounds flag passed via EXTRA_ARGS overrides the
# template default of 150 (argparse uses the last occurrence of a flag).
#
# Submits 6 SLURM jobs:
#
#   Algorithms × Seeds                = 6 jobs
#   --------------------------------
#   {FedAvg, FedProx} × {42, 123, 456}
#
# All on L3 (two_client_70_30_rare_enriched) partition, E=20, 250 rounds,
# μ=0.01 for FedProx. Walk-clock budget: at 4060s for 150 rounds (FedProx
# at L4 on A100), 250 rounds ≈ 6,800s ≈ 1.9 h — comfortably within the
# template's 8 h cap.
#
# Outputs land in a separate directory so they don't clobber the
# existing 150-round L3 result at heterogeneity_ladder/L3_*:
#
#   fl_dermamnist/results/extended_rounds_L3/
#       test_at_best_{fedavg,fedprox}_mu{0.0,0.01}_E20_s{42,123,456}.json
#
# Usage
# -----
#   bash fl_dermamnist/scripts/submit_extended_rounds_L3.sh             # submit
#   DRY_RUN=1 bash fl_dermamnist/scripts/submit_extended_rounds_L3.sh   # print only
#
# Decision rule after the 6 jobs return:
#   - If both methods now plateau before round 250 with the same delta
#     → the original L3 finding holds; FedProx is genuinely behind at L3
#   - If FedProx now plateaus higher than at 150 rounds and matches FedAvg
#     → the original L3 finding was an under-training artefact; revise
#   - If neither method plateaus by round 250 → DermaMNIST + 70/30 mixed
#     skew may need a larger compute budget than this thesis can afford
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

LOCAL_EPOCHS=20
NUM_ROUNDS=250
PARTITION=two_client_70_30_rare_enriched
MU_PROX=0.01

SEEDS=(42 123 456)
ALGOS=("fedavg" "fedprox")

DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh"
OUTDIR="fl_dermamnist/results/extended_rounds_L3"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS_SUMMARY=()
FAILED=()

submit_flower() {
    local algo="$1" mu="$2" seed="$3" jobname="$4"
    # --num-rounds 250 in EXTRA_ARGS overrides the template's hard-coded 150
    # (argparse takes the last occurrence of a duplicated flag).
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FLOWER_TPL"
        "$algo" "$mu" "$seed" "$LOCAL_EPOCHS"
        "$OUTDIR" "$PARTITION"
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
echo "Extended-rounds 3-seed L3 (num_rounds = $NUM_ROUNDS)"
echo "  Partition: $PARTITION"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Algos:     ${ALGOS[*]}"
echo "  Total:     $(( ${#SEEDS[@]} * ${#ALGOS[@]} )) jobs"
echo "============================================================"

for algo in "${ALGOS[@]}"; do
    echo ""
    echo "--- $algo ---"
    for seed in "${SEEDS[@]}"; do
        if [ "$algo" = "fedavg" ]; then
            MU=0.0
        else
            MU="$MU_PROX"
        fi
        submit_flower "$algo" "$MU" "$seed" "mn_extL3_${algo}_s${seed}"
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
echo "Outputs in:   $OUTDIR/"
echo ""
echo "After all 6 jobs finish, on the Mac run:"
echo "  python fl_dermamnist/results/thesis_ready/scripts/analyse_extended_rounds_L3.py"
echo "  (compares selected_round and macro-F1 at 250 vs at 150 rounds;"
echo "   tells you whether FedProx's L3 deficit was under-training)"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
