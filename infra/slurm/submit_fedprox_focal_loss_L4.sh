#!/bin/bash
# Phase 3 Experiment P4 — Extend FedProx × loss compositionality 2×3
# with focal loss column.
#
# B2 established that FedProx is INERT on top of weighted-CE
# (interaction = -0.004). To establish this isn't a weighted-CE-specific
# fluke, we add a third loss (Focal-γ2) and re-test the interaction.
#
# Literature gap (DeCaF short paper / paper section target):
#   - The 2026 Confusion-Calibrated CE paper (S095070512600239X) verbally
#     states "FedProx remains class-agnostic" — but doesn't empirically
#     isolate this across loss families
#   - FedLC (arXiv:2209.00189, 2022) tests logit calibration vs FedAvg
#     only — no FedProx interaction
#   - FedIIC (MICCAI 2023, doi 10.1007/978-3-031-43895-0_65) addresses
#     class imbalance but uses a specialised loss, not FedProx
#
# Design — 6 new runs (extends B2's 12 to 2×3 = 18 total):
#
#   Add Focal-γ=2 column to the existing 2×2:
#     {FedAvg, FedProx} × {Focal-γ2} × 3 seeds = 6 jobs
#
#   Combined with existing B2:
#     {FedAvg, FedProx} × {CE, weighted-CE, Focal-γ2} × 3 seeds = 18 runs
#
# Predicted outcome:
#   - Focal-γ2 alone provides a benefit comparable to weighted-CE
#   - FedProx interaction with Focal-γ2 is also ~0
#   → "FedProx is functionally inert atop any class-imbalance-aware loss"
#     (publishable as a clean negative result across loss families)
#
# Cost: ~6 GPU-h (6 runs × ~60 min). Runs in the same dir as B2.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
NUM_ROUNDS=150
MU_PROX=0.01
FOCAL_GAMMA=2.0
PARTITION=two_client_90_10_rare_stress
SEEDS=(42 123 456)
DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/infra/slurm/slurm_template_flower.sh"
# Same output dir as B2 — the filename-collision fix means new files
# will have a "_loss-focal" tag distinct from CE / weighted-CE
OUTDIR="fl_dermamnist/results/fedprox_weighted_ce_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS=()
FAILED=()

submit() {
    local algo="$1" mu="$2" seed="$3" jobname="$4"
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms --loss-type focal --focal-gamma $FOCAL_GAMMA"
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
echo "P4 — FedProx × focal-loss extension of B2"
echo "  2 algos × focal × 3 seeds = 6 new jobs"
echo "  Extends 2×2 to 2×3 with Focal-γ2 column"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo "" && echo "--- seed $seed ---"
    submit fedavg  0.0       "$seed" "mn_focal_fedavg_s${seed}"
    submit fedprox "$MU_PROX" "$seed" "mn_focal_fedprox_s${seed}"
done

echo "" && echo "Submitted ${#JOBS[@]} jobs (${#FAILED[@]} failed)."
[ ${#FAILED[@]} -ne 0 ] && exit 1
