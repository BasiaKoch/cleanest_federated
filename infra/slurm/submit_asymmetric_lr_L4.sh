#!/bin/bash
# Experiment D1 - Asymmetric learning-rate protocol on L4 (NOVEL extension).
#
# Extends Li 2020 §5.2 from *asymmetric local epochs* to *asymmetric
# learning rates*. Tests whether the FedProx-vs-FedNova mechanism
# distinction (drift-damping vs objective-inconsistency correction)
# extends to LR asymmetry - a regime where Wang 2020's FedNova theory
# does NOT directly apply (FedNova proves correction for unequal τ_i
# only).
#
# Literature gap:
#   - Wang 2020 (FedNova, arXiv:2007.07481) proves correction only
#     for unequal local-step counts τ_i, NOT for asymmetric LRs.
#   - FedLALR (arXiv:2309.09719, 2023) proposes per-client adaptive
#     LRs but does not isolate the FedProx-vs-FedNova mechanism
#     question under LR asymmetry.
#   - FedEff (Nature Sci. Reports 2025, doi:10.1038/s41598-025-22672-1)
#     varies per-client efficiency but the LR-asymmetry FedProx vs
#     FedNova mechanism comparison is absent from the FL literature.
#
# Design - 3 LR-ratio × 3 algorithms × 3 seeds = 27 jobs on L4:
#
#   LR pairs (C0, C1):
#     (0.01, 0.01)   ratio 1:1  - symmetric baseline
#     (0.01, 0.005)  ratio 2:1  - moderate LR asymmetry
#     (0.01, 0.002)  ratio 5:1  - severe LR asymmetry
#
#   Algorithms:
#     FedAvg                 (no drift control)
#     FedProx (μ=0.01)       (drift control via proximal anchor)
#     FedNova                (objective-inconsistency for τ_i, NOT LR)
#
# Partition: L4 (severe 90/10 class skew).
# Seeds: 42, 123, 456.
#
# Predicted outcomes (each is novel if confirmed):
#   - FedProx ABSORBS LR asymmetry: proximal anchor bounds the larger-LR
#     client's drift; performance comparable to LR=1:1 baseline
#   - FedNova COLLAPSES under LR asymmetry: its derivation assumes
#     uniform LR; only corrects for τ_i; predicted ≥0.10 macro-F1 drop
#     at LR ratio 5:1 vs 1:1
#   - FedAvg sits in between (affected but not as severely)
#
# Why this is a contribution:
#   - Cleanly SEPARATES FedNova from FedProx in a regime FedNova
#     wasn't designed for
#   - Maps to a realistic clinical scenario (hospitals with different
#     training infrastructure / GPU memory budgets / batch sizes)
#   - First medical-FL evaluation of the LR-asymmetry FedProx vs
#     FedNova distinction
#
# Cost: ~20 GPU-h (27 runs × ~45 min).
#
# Usage:
#   bash infra/slurm/submit_asymmetric_lr_L4.sh
#   DRY_RUN=1 bash infra/slurm/submit_asymmetric_lr_L4.sh
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
NUM_ROUNDS=150
MU_PROX=0.01
PARTITION=two_client_90_10_rare_stress
SEEDS=(42 123 456)

# LR pairs: (C0_lr, C1_lr) - C0 is the dominant client, C1 the small specialist
# Symmetric baseline + 2 asymmetric ratios
LR_PAIRS=(
    "0.01:0.01"      # 1:1 baseline
    "0.01:0.005"     # 2:1 ratio
    "0.01:0.002"     # 5:1 ratio
)

DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/infra/slurm/slurm_template_flower.sh"
FEDNOVA_TPL="$REPO_ROOT/infra/slurm/slurm_template_fednova.sh"
OUTDIR="fl_dermamnist/results/asymmetric_lr_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS=()
FAILED=()

submit_flower() {
    local algo="$1" mu="$2" lr_spec="$3" seed="$4" jobname="$5"
    # lr_spec is "c0_lr:c1_lr". Parse into the --lr (anchor for non-listed
    # clients) and --lr-per-client (overrides).
    local c0_lr="${lr_spec%:*}"
    local c1_lr="${lr_spec##*:}"
    local lr_per_client_arg=""
    if [ "$c0_lr" != "$c1_lr" ]; then
        lr_per_client_arg="--lr-per-client 0:${c0_lr},1:${c1_lr}"
    fi
    # Pass the dominant-client lr as the global --lr so any non-listed
    # clients (none here at n=2) would use it.
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms --lr $c0_lr $lr_per_client_arg"
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

submit_fednova() {
    local lr_spec="$1" seed="$2" jobname="$3"
    local c0_lr="${lr_spec%:*}"
    local c1_lr="${lr_spec##*:}"
    local lr_per_client_arg=""
    if [ "$c0_lr" != "$c1_lr" ]; then
        lr_per_client_arg="--lr-per-client 0:${c0_lr},1:${c1_lr}"
    fi
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms --lr $c0_lr $lr_per_client_arg"
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FEDNOVA_TPL"
        "$seed" "$E_MAX"
        "$OUTDIR" "$PARTITION"
        "uniform"
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
echo "Experiment D1 — Asymmetric LR on L4 (NOVEL extension of Li 2020 §5.2)"
echo "  3 LR ratios × 3 algos × 3 seeds = 27 jobs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo "" && echo "--- seed $seed ---"
    for lr_pair in "${LR_PAIRS[@]}"; do
        ratio_tag=$(echo "$lr_pair" | tr ':' '-')
        submit_flower fedavg  0.0       "$lr_pair" "$seed" "mn_aLR_fedavg_${ratio_tag}_s${seed}"
        submit_flower fedprox "$MU_PROX" "$lr_pair" "$seed" "mn_aLR_fedprox_${ratio_tag}_s${seed}"
        submit_fednova               "$lr_pair" "$seed" "mn_aLR_fednova_${ratio_tag}_s${seed}"
    done
done

echo "" && echo "Submitted ${#JOBS[@]} jobs (${#FAILED[@]} failed)."
[ ${#FAILED[@]} -ne 0 ] && exit 1
