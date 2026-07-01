#!/bin/bash
# Phase 3 Experiment P1 - FedNova LR-invariance dose-response.
#
# Extends D1 with 3 extreme LR ratios (10:1, 20:1, 50:1) on L4 to:
#   (a) find the breaking point of FedNova's LR-asymmetry absorption
#   (b) test whether the absorption is robust across a 50x dynamic range
#   (c) provide empirical support for the algebraic argument that
#       FedNova's τ_i-normalization produces LR-invariant per-client
#       aggregation
#
# Combined with existing D1 (ratios 1:1, 2:1, 5:1), this gives a clean
# dose-response curve across 6 ratios.
#
# Literature gap (DeCaF MICCAI workshop target):
#   - Wang 2020 (FedNova, arXiv:2007.07481) proves correction only for
#     unequal τ_i; never instantiates per-client LR as a heterogeneity axis
#   - FedACS (arXiv:2505.11304, 2025): explicitly enumerates τ_i, dataset
#     size, compute - but NOT per-client LR
#   - FedLALR (arXiv:2309.09719, 2023): proposes per-client adaptive LR
#     but doesn't benchmark FedAvg/FedProx/FedNova reaction
#
# Design - 3 new LR ratios × 3 algos × 3 seeds = 27 jobs on L4:
#
#   New LR pairs (C0:C1):
#     (0.01, 0.001)   ratio 10:1
#     (0.01, 0.0005)  ratio 20:1
#     (0.01, 0.0002)  ratio 50:1
#
#   Algorithms: FedAvg, FedProx (μ=0.01), FedNova
#   Seeds: 42, 123, 456
#
# Expected ~22 GPU-h (27 × ~45 min). Output files use the lr_per_client
# tag (the filename-collision bug is now fixed in run_one_*.py).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
NUM_ROUNDS=150
MU_PROX=0.01
PARTITION=two_client_90_10_rare_stress
SEEDS=(42 123 456)

# 3 NEW extreme LR pairs (existing D1 covered 1:1, 2:1, 5:1)
LR_PAIRS=(
    "0.01:0.001"     # 10:1 ratio
    "0.01:0.0005"    # 20:1 ratio
    "0.01:0.0002"    # 50:1 ratio
)

DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/infra/slurm/slurm_template_flower.sh"
FEDNOVA_TPL="$REPO_ROOT/infra/slurm/slurm_template_fednova.sh"
# Same output directory as D1 - additive
OUTDIR="fl_dermamnist/results/asymmetric_lr_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS=()
FAILED=()

submit_flower() {
    local algo="$1" mu="$2" lr_spec="$3" seed="$4" jobname="$5"
    local c0_lr="${lr_spec%:*}"
    local c1_lr="${lr_spec##*:}"
    local lr_per_client_arg="--lr-per-client 0:${c0_lr},1:${c1_lr}"
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
    local lr_per_client_arg="--lr-per-client 0:${c0_lr},1:${c1_lr}"
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
echo "P1 — FedNova LR-invariance dose-response (extreme ratios)"
echo "  3 new ratios × 3 algos × 3 seeds = 27 jobs"
echo "  Augments existing D1 (1:1, 2:1, 5:1) for 6-point dose curve"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo "" && echo "--- seed $seed ---"
    for lr_pair in "${LR_PAIRS[@]}"; do
        ratio_tag=$(echo "$lr_pair" | tr ':' '-')
        submit_flower fedavg  0.0       "$lr_pair" "$seed" "mn_lrE_fedavg_${ratio_tag}_s${seed}"
        submit_flower fedprox "$MU_PROX" "$lr_pair" "$seed" "mn_lrE_fedprox_${ratio_tag}_s${seed}"
        submit_fednova               "$lr_pair" "$seed" "mn_lrE_fednova_${ratio_tag}_s${seed}"
    done
done

echo "" && echo "Submitted ${#JOBS[@]} jobs (${#FAILED[@]} failed)."
[ ${#FAILED[@]} -ne 0 ] && exit 1
