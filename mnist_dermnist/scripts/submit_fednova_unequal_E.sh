#!/bin/bash
# Experiment 3 — FedNova × equal-vs-unequal local epochs (mechanism toggle).
#
# Separates two mechanisms that "FedProx helps" usually conflates:
#   (i) drift damping via the proximal term (FedProx)
#  (ii) objective-inconsistency correction under unequal local work (FedNova)
#
# Canonical setup: the original FedNova paper (Wang et al. 2020, NeurIPS,
# arXiv:2007.07481, §5, Figs. 3-4 and Fig. 6) runs exactly this ablation,
# sweeping τ_i across clients. NIID-Bench (Li, Diao, Chen, He 2022,
# arXiv:2102.02079, §4.2) treats {FedAvg, FedProx, FedNova} ×
# {equal-τ, unequal-τ} as the standard four-way comparison. FedShuffle
# (Horváth et al. 2022, arXiv:2204.13169) re-frames this control
# theoretically: FedNova's gains vanish under equal local work, providing
# the mechanism-isolation rationale.
#
# Submits 12 SLURM jobs:
#
#   Levels × Algorithms × Work-regime           = 12 jobs
#   -----------------------------------------
#   {L3, L4} × {FedAvg, FedProx, FedNova} × {equal-E, unequal-E}
#
# Unequal-E setup: Client 0 (large/dominant) does E=20 local epochs;
# Client 1 (small) does E=5. Implemented via the existing system-het
# scaffold (--system-het-mode fixed_stragglers --fixed-straggler-ids 1
# --straggler-epochs 5). This is the "stragglers" semantics of FedAvg
# (Li et al. 2020 §5.2) and the unequal-τ_i setting of FedNova.
#
# Equal-E baseline: --system-het-mode uniform with E=20 for both clients.
# At seed 42 we already have equal-E data in heterogeneity_ladder/L{3,4}_*;
# we re-submit the equal-E arm here so the comparison is on matched
# compute (similar nodes / same code state), addressing the cross-node
# variance issue documented in node_pinned_L4.
#
# Single seed (42), Stage A pilot. Promote to 3 seeds only if the
# equal-vs-unequal toggle produces a clear differential.
#
# Outputs land in a dedicated directory:
#   mnist_dermnist/results/fednova_unequal_E/L{3,4}_<partition>/
#       test_at_best_{algo}_..._s42.json
#
# Filenames distinguish equal-E (no _sh- tag) from unequal-E
# (_sh-fixed_stragglers tag) automatically via run_one_flower.py's
# existing sh_tag logic.
#
# Usage
# -----
#   bash mnist_dermnist/scripts/submit_fednova_unequal_E.sh             # submit
#   DRY_RUN=1 bash mnist_dermnist/scripts/submit_fednova_unequal_E.sh   # print only
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

SEED=42
E_MAX=20
E_STRAGGLER=5
NUM_ROUNDS=150
MU_PROX=0.01

FLOWER_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh"
FEDNOVA_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_fednova.sh"

# 2 ladder levels.
LEVELS=("L3" "L4")
PARTITIONS=(
    "two_client_70_30_rare_enriched"
    "two_client_90_10_rare_stress"
)

DRY_RUN="${DRY_RUN:-0}"

JOBS_SUMMARY=()
FAILED=()

submit_flower() {
    # FedAvg / FedProx flower runner
    local algo="$1" mu="$2" partition="$3" outdir="$4" jobname="$5" sh_mode="$6"
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms"
    if [ "$sh_mode" = "fixed_stragglers" ]; then
        extra="$extra --system-het-mode fixed_stragglers --fixed-straggler-ids 1 --straggler-epochs $E_STRAGGLER"
    fi
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FLOWER_TPL"
        "$algo" "$mu" "$SEED" "$E_MAX"
        "$outdir" "$partition"
        "$extra")
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
    local partition="$1" outdir="$2" jobname="$3" sh_mode="$4"
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms"
    if [ "$sh_mode" = "fixed_stragglers" ]; then
        extra="$extra --fixed-straggler-ids 1 --straggler-epochs $E_STRAGGLER"
    fi
    # FedNova template signature: seed | E_max | out_dir | partition | sh_mode | extra
    local cmd=(sbatch --parsable
        --job-name="$jobname"
        "$FEDNOVA_TPL"
        "$SEED" "$E_MAX"
        "$outdir" "$partition"
        "$sh_mode"
        "$extra")
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
echo "Experiment 3 — FedNova × equal-vs-unequal local epochs"
echo "  Seed: $SEED  (Stage A pilot; promote to 3 seeds if differential)"
echo "  Levels: ${LEVELS[*]}"
echo "  Algos:  fedavg, fedprox(μ=$MU_PROX), fednova"
echo "  Work:   equal (E_C0=$E_MAX, E_C1=$E_MAX)"
echo "          unequal (E_C0=$E_MAX, E_C1=$E_STRAGGLER)"
echo "  Total:  12 jobs"
echo "============================================================"

for i in "${!LEVELS[@]}"; do
    LEVEL=${LEVELS[$i]}
    PARTITION=${PARTITIONS[$i]}
    OUTDIR="mnist_dermnist/results/fednova_unequal_E/${LEVEL}_${PARTITION}"
    mkdir -p "$REPO_ROOT/$OUTDIR"

    echo ""
    echo "--- $LEVEL: $PARTITION ---"

    for SH_MODE in "uniform" "fixed_stragglers"; do
        if [ "$SH_MODE" = "uniform" ]; then
            REGIME_TAG="eq"
        else
            REGIME_TAG="uneq"
        fi
        submit_flower fedavg  0.0       "$PARTITION" "$OUTDIR" \
            "mn_fnueE_${LEVEL}_fedavg_${REGIME_TAG}_s${SEED}" "$SH_MODE"
        submit_flower fedprox "$MU_PROX" "$PARTITION" "$OUTDIR" \
            "mn_fnueE_${LEVEL}_fedprox_${REGIME_TAG}_s${SEED}" "$SH_MODE"
        submit_fednova "$PARTITION" "$OUTDIR" \
            "mn_fnueE_${LEVEL}_fednova_${REGIME_TAG}_s${SEED}" "$SH_MODE"
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
echo "Outputs in:   mnist_dermnist/results/fednova_unequal_E/L{3,4}_*/"
echo ""
echo "After all 12 jobs finish, on the Mac run:"
echo "  python mnist_dermnist/results/thesis_ready/scripts/analyse_fednova_unequal_E.py"
echo "  (decomposes drift-damping (FedProx) vs objective-inconsistency"
echo "   correction (FedNova) by toggling equal/unequal local epochs)"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
