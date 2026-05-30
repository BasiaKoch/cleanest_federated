#!/bin/bash
# Heterogeneity-ladder STAGE A pilot — one seed, all five levels.
#
# Submits 13 SLURM jobs (or fewer with --dry-run / SKIP_FEDNOVA):
#
#   Levels  ×  Methods                        =  jobs
#   ---------------------------------------------------
#   L0..L4  ×  {FedAvg, FedProx}              =  10
#   L1,L3,L4 ×  FedNova (diagnostic for       =   3
#                unequal local-work skew)
#                                              -----
#                                              13
#
# Purpose: confirm the ladder behaves sensibly (FedProx ≈ FedAvg at L0/L1,
# any advantage emerges at L2+, FedNova clarifies whether unequal local
# work explains it) BEFORE running Stage B at 3 seeds.
#
# All Stage A runs reuse the existing flower / fednova SLURM templates
# and the existing seed convention (seed 42).
#
# Usage
# -----
#   bash mnist_dermnist/scripts/submit_ladder_pilot.sh             # submit
#   DRY_RUN=1 bash mnist_dermnist/scripts/submit_ladder_pilot.sh   # print only
#   SKIP_FEDNOVA=1 bash mnist_dermnist/scripts/submit_ladder_pilot.sh
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

SEED=42
MU_PROX=0.01
LOCAL_EPOCHS=20
NUM_ROUNDS=150

FLOWER_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh"
FEDNOVA_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_fednova.sh"

# Ladder partitions (parallel arrays of level number and partition name).
LEVELS=(0 1 2 3 4)
PARTITIONS=(
    two_client_50_50_stratified_iid
    two_client_86_14_quantity_only_stratified
    two_client_50_50_label_skew_only
    two_client_70_30_rare_enriched
    two_client_90_10_rare_stress
)

# FedNova is only run on levels where unequal local-work could plausibly
# matter (i.e. the asymmetric levels). On L0 and L2 the two clients
# already use the same E, so FedNova reduces to FedAvg.
FEDNOVA_LEVELS=(1 3 4)

DRY_RUN="${DRY_RUN:-0}"
SKIP_FEDNOVA="${SKIP_FEDNOVA:-0}"

DRY_PREFIX=""
if [ "$DRY_RUN" = "1" ]; then
    DRY_PREFIX="echo DRY-RUN:"
fi

JOBS_SUMMARY=()
FAILED=()

submit_flower() {
    local algo="$1" mu="$2" partition="$3" outdir="$4" jobname="$5"
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FLOWER_TPL"
        "$algo" "$mu" "$SEED" "$LOCAL_EPOCHS"
        "$outdir" "$partition"
        "--num-rounds $NUM_ROUNDS --log-update-norms --save-best-checkpoint")
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

submit_fednova() {
    local partition="$1" outdir="$2" jobname="$3"
    # FedNova has its own runner+template (no mu, no algorithm flag).
    # Template args:  seed | local_epochs | out_dir | partition |
    #                 system_het_mode | extra_args
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FEDNOVA_TPL"
        "$SEED" "$LOCAL_EPOCHS"
        "$outdir" "$partition"
        "uniform"
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
echo "Heterogeneity-ladder Stage A pilot (seed $SEED)"
echo "============================================================"

for i in "${!LEVELS[@]}"; do
    LEVEL=${LEVELS[$i]}
    PARTITION=${PARTITIONS[$i]}
    OUTDIR="mnist_dermnist/results/heterogeneity_ladder/L${LEVEL}_${PARTITION}"
    mkdir -p "$REPO_ROOT/$OUTDIR"

    echo ""
    echo "--- Level $LEVEL: $PARTITION ---"

    # FedAvg
    submit_flower fedavg  0.0 "$PARTITION" "$OUTDIR" \
        "mn_ladder_L${LEVEL}_fedavg_s${SEED}"

    # FedProx
    submit_flower fedprox "$MU_PROX" "$PARTITION" "$OUTDIR" \
        "mn_ladder_L${LEVEL}_fedprox_s${SEED}"

    # FedNova on the asymmetric levels only.
    if [ "$SKIP_FEDNOVA" = "1" ]; then
        continue
    fi
    for fn_level in "${FEDNOVA_LEVELS[@]}"; do
        if [ "$fn_level" = "$LEVEL" ]; then
            submit_fednova "$PARTITION" "$OUTDIR" \
                "mn_ladder_L${LEVEL}_fednova_s${SEED}"
            break
        fi
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
echo "Outputs in:   mnist_dermnist/results/heterogeneity_ladder/L<level>_<partition>/"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
