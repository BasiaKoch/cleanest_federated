#!/bin/bash
# Node-pinned 3-seed L4 — variance isolation experiment.
#
# Tests whether the L4 FedProx-vs-FedAvg deficit observed at single seed
# (FedAvg 0.518, FedProx 0.492, Δ = -0.026) is a real algorithmic effect
# or an artefact of cross-node CUDA non-determinism. The original ladder
# pilot ran FedAvg on gpu-q-13 and FedProx on gpu-q-40 — different
# physical GPUs, same nominal seed. The cross-run macro-F1 spread on
# seed 42 across HPC nodes is ≈ 0.04, larger than the algorithm gap.
#
# Submits 6 SLURM jobs, all pinned to the SAME compute node:
#
#   Algorithms × Seeds                = 6 jobs
#   --------------------------------
#   {FedAvg, FedProx} × {42, 123, 456}
#
# All runs use the L4 (two_client_90_10_rare_stress) partition, the
# default thesis hyperparameters (E=20, 150 rounds, μ=0.01 for FedProx),
# and the existing slurm_template_flower.sh.
#
# Decision rule after the 6 jobs return:
#   - If the FedAvg-vs-FedProx gap > seed spread          → real effect, commit to Stage B
#   - If the gap is within seed spread                    → algorithms are tied, write that
#   - If FedProx now wins / loses cleanly across all 3    → revisit the ladder narrative
#
# Outputs land in a separate directory so they don't clobber the
# existing single-seed L4 result at heterogeneity_ladder/L4_*:
#
#   fl_dermamnist/results/node_pinned_L4/
#       test_at_best_{fedavg,fedprox}_mu{0.0,0.01}_E20_s{42,123,456}.json
#
# Usage
# -----
#   bash fl_dermamnist/scripts/submit_node_pinned_L4.sh                        # default node gpu-q-13
#   NODELIST=gpu-q-40 bash fl_dermamnist/scripts/submit_node_pinned_L4.sh      # override node
#   DRY_RUN=1 bash fl_dermamnist/scripts/submit_node_pinned_L4.sh              # print only
#
# Note on the node choice: gpu-q-13 was where the original FedAvg run
# succeeded at L4. If that node is congested or unavailable, point
# NODELIST at any single ampere node — the experiment's only requirement
# is that all 6 jobs share the same physical hardware.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

LOCAL_EPOCHS=20
NUM_ROUNDS=150
PARTITION=two_client_90_10_rare_stress
MU_PROX=0.01

SEEDS=(42 123 456)
ALGOS=("fedavg" "fedprox")

NODELIST="${NODELIST:-gpu-q-13}"
DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh"
OUTDIR="fl_dermamnist/results/node_pinned_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS_SUMMARY=()
FAILED=()

submit_flower() {
    local algo="$1" mu="$2" seed="$3" jobname="$4"
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        --nodelist="$NODELIST"
        "$FLOWER_TPL"
        "$algo" "$mu" "$seed" "$LOCAL_EPOCHS"
        "$OUTDIR" "$PARTITION"
        "--num-rounds $NUM_ROUNDS --log-update-norms")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY-RUN: ${cmd[*]}"
        echo "  -> would submit  $jobname  on  $NODELIST"
        return
    fi
    local jid
    if ! jid=$("${cmd[@]}"); then
        echo "  FAILED: $jobname" >&2
        FAILED+=("$jobname")
        return 1
    fi
    JOBS_SUMMARY+=("$jid  $jobname  ($NODELIST)")
    echo "  $jid  $jobname  on  $NODELIST"
}

echo "============================================================"
echo "Node-pinned 3-seed L4 variance isolation (node: $NODELIST)"
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
        submit_flower "$algo" "$MU" "$seed" "mn_pinL4_${algo}_s${seed}"
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
echo "  python fl_dermamnist/results/thesis_ready/scripts/analyse_node_pinned_L4.py"
echo "  (produces mean ± SD per algorithm with all 3 seeds on same node;"
echo "   if SD spans the FedAvg/FedProx gap → algorithm difference is noise)"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
