#!/bin/bash
# Phase 3 Experiment P2 — Cross-partition replication of Li 2020 §5.2
# decomposition on L1 (quantity-only skew, JS = 0.0).
#
# The original Li 2020 §5.2 4-condition factorial was run on L4 (severe
# 90/10 class-disjoint). To establish cross-partition INVARIANCE of the
# γ-inexact-attribution mechanism, we replicate the SAME factorial on
# L1 (quantity 86/14 but no class skew — every class on both clients).
#
# Literature gap (MIDL / MELBA target):
#   - No 2024-2026 paper repeats Li 2020 §5.2 decomposition with the
#     Condition-4 control
#   - Frontiers 2025 survey (10.3389/fcomp.2025.1617597) explicitly notes
#     FedProx vs FedAvg results are "protocol-confounded across literature"
#   - NIID-Bench (Li 2022, arXiv:2102.02079) does not separate γ-inexact
#     handling from proximal anchoring
#
# Design — 4 conditions × 3 seeds on L1 = 12 jobs:
#
#   #  Algorithm      Stragglers      Drop?       Role
#   ---------------------------------------------------------------
#   1  FedAvg         none (E=20)     n/a         L1 baseline
#   2  FedAvg         C1 = E=5        DROP        Li §5.2 FA arm
#   3  FedProx μ=0.01 C1 = E=5        KEEP        Li §5.2 FP arm
#   4  FedProx μ=0.01 C1 = E=5        DROP        Control
#
# Predicted outcome (binary):
#   IF the γ-inexact attribution survives on L1 (same Conditional-4 isolation):
#     → mechanism is robust ACROSS partition types → strong cross-partition
#       claim, publishable as "γ-inexact handling explains FedProx advantage
#       on both label-skew (L4) AND quantity-skew (L1) heterogeneity"
#   IF the attribution differs on L1:
#     → mechanism is label-skew-specific → narrower but still publishable
#       finding
#
# Cost: ~8 GPU-h. Pairs with existing L4 data for the cross-partition
# invariance claim.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
E_STRAGGLER=5
NUM_ROUNDS=150
MU_PROX=0.01
PARTITION=two_client_86_14_quantity_only_stratified   # L1 — quantity-only

SEEDS=(42 123 456)
DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/infra/slurm/slurm_template_flower.sh"
OUTDIR="fl_dermamnist/results/li2020_asymmetric_L1"
mkdir -p "$REPO_ROOT/$OUTDIR"

STRAGGLER_ARGS="--system-het-mode fixed_stragglers --fixed-straggler-ids 1 --straggler-epochs $E_STRAGGLER"

JOBS=()
FAILED=()

submit_flower() {
    local algo="$1" mu="$2" seed="$3" jobname="$4" runner_extra="$5"
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms"
    if [ -n "$runner_extra" ]; then
        extra="$extra $runner_extra"
    fi
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FLOWER_TPL"
        "$algo" "$mu" "$seed" "$E_MAX"
        "$OUTDIR" "$PARTITION"
        "$extra")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY-RUN: ${cmd[*]}"
        return
    fi
    local jid
    if ! jid=$("${cmd[@]}"); then
        FAILED+=("$jobname")
        return 1
    fi
    JOBS+=("$jid  $jobname")
    echo "  $jid  $jobname"
}

echo "============================================================"
echo "P2 — Li 2020 §5.2 cross-partition replication on L1"
echo "  Partition: $PARTITION  (quantity-only 86/14, JS=0)"
echo "  4 conditions × 3 seeds = 12 jobs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo "" && echo "--- seed $seed ---"
    # 1. FedAvg baseline (no straggler)
    submit_flower fedavg 0.0 "$seed" "mn_L1L20_fedavg_baseline_s${seed}" ""
    # 2. FedAvg + drop + straggler
    submit_flower fedavg 0.0 "$seed" "mn_L1L20_fedavg_drop_s${seed}" \
        "$STRAGGLER_ARGS --drop-stragglers"
    # 3. FedProx + γ-inexact + straggler
    submit_flower fedprox "$MU_PROX" "$seed" "mn_L1L20_fedprox_inexact_s${seed}" \
        "$STRAGGLER_ARGS"
    # 4. FedProx + drop control
    submit_flower fedprox "$MU_PROX" "$seed" "mn_L1L20_fedprox_drop_ctrl_s${seed}" \
        "$STRAGGLER_ARGS --drop-stragglers"
done

echo "" && echo "Submitted ${#JOBS[@]} jobs (${#FAILED[@]} failed)."
[ ${#FAILED[@]} -ne 0 ] && exit 1
