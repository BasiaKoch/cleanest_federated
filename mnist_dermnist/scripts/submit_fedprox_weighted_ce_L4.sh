#!/bin/bash
# Experiment B2 — FedProx × class-weighted-CE 2×2 ablation on L4.
#
# Tests whether FedProx and inverse-frequency class-weighted CE are
# SUBSTITUTES, COMPLEMENTS, or NON-ADDITIVE on partition-induced
# class imbalance. The class-imbalance-FL survey (arXiv:2303.11673,
# 2023) and FedLC (arXiv:2209.00189, 2022) compare logit-calibration
# vs FedAvg but never test FedProx × weighted-CE compositionality
# on partition-induced imbalance.
#
# Design — 2×2 × 3 seeds = 12 jobs:
#
#   Algorithm × Loss
#   FedAvg  + standard CE          (baseline)
#   FedAvg  + class_weighted_ce    (loss-only intervention)
#   FedProx + standard CE          (algorithm-only intervention)
#   FedProx + class_weighted_ce    (combined)
#
# Partition: L4 (severe 90/10 class skew). Seeds: 42, 123, 456.
#
# Predicted outcomes (any of these would be a contribution):
#   - SUBSTITUTES: combined ≈ best-of-either → choose the cheaper one
#   - COMPLEMENTS (additive): combined > either alone, sum of effects
#   - NON-ADDITIVE: combined ≠ sum (most interesting; counters "stacking")
#
# Cost: ~10 GPU-h.
#
# Usage:
#   bash mnist_dermnist/scripts/submit_fedprox_weighted_ce_L4.sh
#   DRY_RUN=1 bash mnist_dermnist/scripts/submit_fedprox_weighted_ce_L4.sh
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
NUM_ROUNDS=150
MU_PROX=0.01
PARTITION=two_client_90_10_rare_stress
SEEDS=(42 123 456)
DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh"
OUTDIR="mnist_dermnist/results/fedprox_weighted_ce_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS=()
FAILED=()

submit() {
    local algo="$1" mu="$2" loss="$3" seed="$4" jobname="$5"
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms --loss-type $loss"
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
echo "Experiment B2 — FedProx × weighted-CE 2×2 on L4"
echo "  3 seeds × 2 algos × 2 losses = 12 jobs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo "" && echo "--- seed $seed ---"
    submit fedavg  0.0       ce                  "$seed" "mn_wce_fedavg_ce_s${seed}"
    submit fedavg  0.0       class_weighted_ce   "$seed" "mn_wce_fedavg_wce_s${seed}"
    submit fedprox "$MU_PROX" ce                  "$seed" "mn_wce_fedprox_ce_s${seed}"
    submit fedprox "$MU_PROX" class_weighted_ce   "$seed" "mn_wce_fedprox_wce_s${seed}"
done

echo "" && echo "Submitted ${#JOBS[@]} jobs (${#FAILED[@]} failed)."
[ ${#FAILED[@]} -ne 0 ] && exit 1
