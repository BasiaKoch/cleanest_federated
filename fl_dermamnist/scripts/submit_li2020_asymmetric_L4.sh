#!/bin/bash
# Experiment 10 — Li 2020 §5.2 asymmetric protocol on L4 ("FedProx wins" setting).
#
# This experiment reproduces the conditions under which Li et al. 2020
# (FedProx, MLSys, arXiv:1812.06127) explicitly claim FedProx wins by
# ~22% over FedAvg. The current thesis data is FedProx-neutral because
# we have only run the SYMMETRIC protocol (both algorithms see the same
# client updates). Li 2020 §5.2 is specifically the ASYMMETRIC protocol:
# FedAvg drops stragglers, FedProx keeps them via γ-inexact local
# updates whose drift is bounded by the proximal anchor.
#
# Setup on L4 (severe class-disjoint heterogeneity, JS = 0.385):
#   - Client 0: dominant, 90% of data, common classes only. E = 20.
#   - Client 1: small specialist, 10% of data, all rare classes (dermato,
#               melanoma, vascular). E_straggler = 5 (cannot finish E_max).
#
# Asymmetric protocol (Li 2020 §5.2):
#   - FedAvg arm: --drop-stragglers → Client 1's update is DISCARDED every
#                 round → FedAvg trains ONLY on Client 0's data → it
#                 literally never sees rare-class examples → rare-class
#                 F1 should collapse to ~0.
#   - FedProx arm: no --drop-stragglers → Client 1's γ-inexact (partial)
#                  update is INCLUDED in aggregation. The proximal anchor
#                  (μ/2)‖w − w^t‖² stabilises the partial update so it
#                  doesn't push the global model off-manifold. Rare-class
#                  signal survives.
#
# Design — 4 conditions × 3 seeds = 12 jobs:
#
#   #  Algorithm      Stragglers    Drop?   Why
#   ----------------------------------------------------------------
#   1  FedAvg         none (E=20)   n/a     Baseline (current thesis #)
#   2  FedAvg         C1=5          DROP    Li 2020 §5.2 FedAvg arm
#   3  FedProx μ=0.01 C1=5          KEEP    Li 2020 §5.2 FedProx arm
#   4  FedProx μ=0.01 C1=5          DROP    Control — isolates protocol
#                                            from algorithm
#
# Headline reading: gap (2 → 3) is the Li 2020 §5.2 protocol effect on
# this task. Predicted to be large because L4 is class-disjoint and
# dropping Client 1 removes all rare-class training signal.
#
# Why this is academically defensible:
#   - It IS Li 2020's §5.2 protocol, not a contrived setup.
#   - The failure mode is mechanically obvious (FedAvg can't learn classes
#     it never sees).
#   - Condition 4 controls for "protocol vs algorithm" — if 3 ≫ 4, the
#     advantage is from γ-inexact handling (FedProx's mechanism), not the
#     algorithm name.
#   - Baseline condition 1 anchors to existing thesis hyperparameters.
#
# Citation: Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A.,
# Smith, V. "Federated Optimization in Heterogeneous Networks" §5.2
# "Lessons Learned", MLSys 2020. arXiv:1812.06127.
#
# Outputs:
#   fl_dermamnist/results/li2020_asymmetric_L4/
#       test_at_best_fedavg_mu0.0_E20_sh-fixed_stragglers[_drop]_s{42,123,456}.json
#       test_at_best_fedprox_mu0.01_E20_sh-fixed_stragglers[_drop]_s{42,123,456}.json
#       test_at_best_fedavg_mu0.0_E20_s{42,123,456}.json   (baseline, no sh tag)
#
# Usage
# -----
#   bash fl_dermamnist/scripts/submit_li2020_asymmetric_L4.sh             # 12 jobs
#   DRY_RUN=1 bash fl_dermamnist/scripts/submit_li2020_asymmetric_L4.sh   # print only
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
E_STRAGGLER=5
NUM_ROUNDS=150
MU_PROX=0.01
PARTITION=two_client_90_10_rare_stress

SEEDS=(42 123 456)

DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh"
OUTDIR="fl_dermamnist/results/li2020_asymmetric_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS_SUMMARY=()
FAILED=()

# Common straggler config (Li 2020 §5.2 with fixed C1 straggler).
STRAGGLER_ARGS="--system-het-mode fixed_stragglers --fixed-straggler-ids 1 --straggler-epochs $E_STRAGGLER"

submit_flower() {
    # algo | mu | seed | jobname | extra_runner_args
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
echo "Experiment 10 — Li 2020 §5.2 asymmetric protocol on L4"
echo "  Partition: $PARTITION  (class-disjoint 90/10)"
echo "  E_max:     $E_MAX   E_straggler: $E_STRAGGLER (fixed: client 1)"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Conditions: 4 (baseline, FedAvg+drop, FedProx-γ-inexact, FedProx+drop control)"
echo "  Total:     12 jobs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- seed $seed ---"

    # 1. FedAvg baseline (no stragglers, no drop). Anchors to existing thesis.
    submit_flower fedavg 0.0 "$seed" "mn_li20_L4_fedavg_baseline_s${seed}" ""

    # 2. FedAvg + drop-stragglers + fixed C1 straggler.
    #    Li 2020 §5.2 FedAvg arm: discards Client 1's update every round.
    submit_flower fedavg 0.0 "$seed" \
        "mn_li20_L4_fedavg_drop_s${seed}" \
        "$STRAGGLER_ARGS --drop-stragglers"

    # 3. FedProx (γ-inexact) + fixed C1 straggler.
    #    Li 2020 §5.2 FedProx arm: keeps Client 1's partial update; the
    #    proximal anchor stabilises the inexact local solution.
    submit_flower fedprox "$MU_PROX" "$seed" \
        "mn_li20_L4_fedprox_inexact_s${seed}" \
        "$STRAGGLER_ARGS"

    # 4. FedProx + drop-stragglers + fixed C1 straggler.
    #    CONTROL: isolates protocol effect from algorithm effect.
    #    If 3 ≫ 4, the win comes from γ-inexact handling.
    submit_flower fedprox "$MU_PROX" "$seed" \
        "mn_li20_L4_fedprox_drop_control_s${seed}" \
        "$STRAGGLER_ARGS --drop-stragglers"
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
echo "After all 12 jobs finish, on the Mac run:"
echo "  python fl_dermamnist/results/thesis_ready/scripts/analyse_li2020_asymmetric_L4.py"
echo ""
echo "Expected headline (predicted from Li 2020 §5.2):"
echo "  Condition 1 (FedAvg baseline):       macro-F1 ≈ 0.52"
echo "  Condition 2 (FedAvg + drop):         macro-F1 collapses; rare-class F1 → ~0"
echo "  Condition 3 (FedProx + γ-inexact):   macro-F1 close to baseline; rare-class preserved"
echo "  Condition 4 (FedProx + drop ctrl):   between (2) and (3); confirms protocol matters"
echo "  → Headline gap (2 → 3) is the Li 2020 §5.2 result on DermaMNIST."
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
