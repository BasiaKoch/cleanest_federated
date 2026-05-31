#!/bin/bash
# Re-submit ONLY the missing FedNova asymmetric-LR jobs for D1.
#
# The original submit_asymmetric_lr_L4.sh ran 27 jobs but the FedNova
# runner had a filename-collision bug: stem didn't include the
# lr_per_client tag, so all 3 FedNova LR-ratios per seed overwrote
# each other, leaving only 1 file per seed (the last to finish).
#
# Bug fix is now in run_one_fednova_flower.py (commit to be pushed).
# This script re-submits ONLY the 6 missing asymmetric-LR FedNova
# jobs (2 ratios × 3 seeds). The symmetric (1:1) FedNova files
# already exist and are valid; this script does NOT touch them.
#
# Cost: ~5 GPU-h.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
NUM_ROUNDS=150
PARTITION=two_client_90_10_rare_stress
SEEDS=(42 123 456)
ASYM_LR_PAIRS=(
    "0.01:0.005"     # 2:1 ratio
    "0.01:0.002"     # 5:1 ratio
)
DRY_RUN="${DRY_RUN:-0}"

FEDNOVA_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_fednova.sh"
OUTDIR="mnist_dermnist/results/asymmetric_lr_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS=()
FAILED=()

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
echo "D1 fix: re-submit 6 missing FedNova asymmetric-LR jobs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    for lr_pair in "${ASYM_LR_PAIRS[@]}"; do
        ratio_tag=$(echo "$lr_pair" | tr ':' '-')
        submit_fednova "$lr_pair" "$seed" "mn_aLR_fednova_${ratio_tag}_s${seed}_v2"
    done
done

echo "" && echo "Submitted ${#JOBS[@]} jobs (${#FAILED[@]} failed)."
[ ${#FAILED[@]} -ne 0 ] && exit 1
