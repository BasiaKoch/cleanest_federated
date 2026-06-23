#!/bin/bash
# Experiment 11 — FedProx "perfect-storm" L4: literature-canonical setting
# in which FedProx is predicted to win by the largest margin.
#
# Motivation. A literature review of the conditions under which FedProx
# beats FedAvg identified that the current thesis configuration is
# silently configured AGAINST FedProx in FIVE specific ways:
#
#   (1) μ = 0.01 (too small).
#       NIID-Bench (Li, Diao, Chen, He 2022, arXiv:2102.02079) §V.B:
#       "Since the best μ is always small, the regularization term in
#       FedProx has little influence on the training. Thus, FedProx
#       and FedAvg usually have similar convergence speed and final
#       accuracy." HeteRo-Select (2025, arXiv:2508.06692) §V.D:
#       "increasing the regularization strength from μ = 0.01 to μ =
#       0.1 had a transformative effect."
#       → Use μ = 1.0 (largest in Li 2020's grid {0.001, 0.01, 0.1, 1}
#       and the value Li 2020 §5.3.2 selected for the most heterogeneous
#       partitions: Synthetic(1,1), MNIST, FEMNIST).
#
#   (2) batch_size = 32 (too large).
#       FedProx paper (Li et al. 2020, arXiv:1812.06127) Appendix C.2:
#       "We use a batch size of 10 for all experiments." Small batch
#       → noisier local SGD → more drift per local epoch → more
#       headroom for the proximal anchor to provide value.
#       → Use batch_size = 10.
#
#   (3) momentum = 0.9 (footgun).
#       Li 2020 §5.1: "we employ SGD as a local solver for FedProx" —
#       plain SGD, no momentum. Momentum compounds with the proximal
#       pull across steps; none of the cited FedProx-favourable papers
#       use it. The Flower reference baseline likewise uses no momentum.
#       → Use momentum = 0.0.
#
#   (4) No straggler stress.
#       The FedProx paper's headline "22% improvement" (Section 5.3.2,
#       Figure 7) is specifically at 90% stragglers. With zero
#       stragglers, the systems-heterogeneity half of FedProx's
#       advantage is unavailable and only the statistical-heterogeneity
#       half remains.
#       → Use random_stragglers mode, fraction = 0.9.
#
#   (5) No asymmetric protocol.
#       Without --drop-stragglers, FedAvg keeps the same partial
#       updates FedProx does — neutralising the protocol asymmetry that
#       drives the 22% headline. Li 2020 §5.2 specifically frames the
#       comparison as: FedAvg drops stragglers (its native protocol);
#       FedProx aggregates partial work (its native γ-inexact protocol).
#       → Add --drop-stragglers to FedAvg only.
#
# Combined, these five changes replicate the Li 2020 Section 5.2 +
# Section 5.3.2 experimental conditions as closely as the 2-client
# DermaMNIST setup allows. The partition (L4: class-disjoint 90/10)
# is the closest available 2-client analogue to NIID-Bench's "#C=1"
# partition (NIID-Bench Table III, FMNIST #C=1: largest transferable
# FedProx-over-FedAvg gap in the published literature at +17.7 pp).
#
# Design — 3 conditions × 3 seeds = 9 jobs on L4:
#
#   Cond | Algorithm | μ    | bs | mom | Stragglers     | Drop?       | Role
#   -----+-----------+------+----+-----+----------------+-------------+----------------------
#    1   | FedAvg    |  —   | 10 | 0.0 | random, f=0.9  | YES (drop)  | Li-2020 FA arm
#    2   | FedProx   | 1.0  | 10 | 0.0 | random, f=0.9  | NO (keep)   | ⭐ Li-2020 FP arm
#    3   | FedProx   | 0.01 | 10 | 0.0 | random, f=0.9  | NO (keep)   | μ ablation
#
# Headline pair (1, 2) is the canonical Li-2020 FedAvg-vs-FedProx
# comparison reproduced on DermaMNIST. Pair (2, 3) isolates the μ
# choice — tells you whether μ = 1.0 vs μ = 0.01 matters in this
# regime, with all other hyperparameters held fixed.
#
# Cross-reference: thesis-baseline data (FedAvg μ=0, FedProx μ=0.01 at
# L4 with batch_size=32, momentum=0.9, no stragglers) is already
# covered by submit_node_pinned_L4.sh. The "configuration gap"
# (perfect-storm minus thesis-baseline) tells you how much of the
# observed FedProx win is due to the configuration change.
#
# Caveats to document in the write-up:
#   (a) DermaMNIST is 28×28 RGB; the largest published gaps (rcv1 in
#       NIID-Bench) are on high-dimensional sparse data and will not
#       transfer in absolute magnitude.
#   (b) The 22% headline averages five datasets; single-dataset gaps
#       are typically smaller (5-18 pp depending on dataset).
#   (c) n = 2 limits variance; published big-gap results have 10-1000
#       clients. The cross-node noise floor (≈ 0.04 macro-F1) still
#       applies.
#
# Cost: ~9 GPU-h (9 runs at ~1 h each; bs=10 has more iterations per
# epoch but smaller per-iter work, net wall-clock ≈ baseline).
#
# Outputs (filenames disambiguated by run_one_flower.py's tag system):
#   fl_dermamnist/results/fedprox_perfect_storm_L4/
#       test_at_best_fedavg_mu0.0_E20_sh-random_stragglers_drop_s{42,123,456}.json
#       test_at_best_fedprox_mu1.0_E20_sh-random_stragglers_s{42,123,456}.json
#       test_at_best_fedprox_mu0.01_E20_sh-random_stragglers_s{42,123,456}.json
#
# Usage
# -----
#   bash infra/slurm/submit_fedprox_perfect_storm_L4.sh            # 9 jobs
#   DRY_RUN=1 bash infra/slurm/submit_fedprox_perfect_storm_L4.sh  # print only
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
E_STRAGGLER=5
STRAGGLER_FRACTION=0.9
NUM_ROUNDS=150
PARTITION=two_client_90_10_rare_stress

# Storm-config hyperparameters (override slurm template defaults via EXTRA_ARGS).
STORM_BATCH=10
STORM_MOMENTUM=0.0

SEEDS=(42 123 456)

DRY_RUN="${DRY_RUN:-0}"

FLOWER_TPL="$REPO_ROOT/infra/slurm/slurm_template_flower.sh"
OUTDIR="fl_dermamnist/results/fedprox_perfect_storm_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS_SUMMARY=()
FAILED=()

# Common straggler + storm-config args (passed via EXTRA_ARGS; argparse takes
# the LAST occurrence of duplicated flags, so these override the template's
# hard-coded --batch-size 32 and default momentum=0.9).
STRAGGLER_ARGS="--system-het-mode random_stragglers --straggler-fraction $STRAGGLER_FRACTION --straggler-epochs $E_STRAGGLER"
STORM_CONFIG_ARGS="--batch-size $STORM_BATCH --momentum $STORM_MOMENTUM"

submit_flower() {
    local algo="$1" mu="$2" seed="$3" jobname="$4" runner_extra="$5"
    local extra="--num-rounds $NUM_ROUNDS --log-update-norms $STORM_CONFIG_ARGS $STRAGGLER_ARGS"
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
echo "Experiment 11 — FedProx PERFECT-STORM L4 (literature-canonical)"
echo "  Partition:        $PARTITION (closest 2-client analogue to NIID-Bench #C=1)"
echo "  E_max:            $E_MAX local epochs"
echo "  Batch size:       $STORM_BATCH (vs thesis default 32 — small-batch FedProx-favouring)"
echo "  Momentum:         $STORM_MOMENTUM (vs thesis default 0.9 — Li 2020 plain SGD)"
echo "  Stragglers:       random, fraction=$STRAGGLER_FRACTION, E_straggler=$E_STRAGGLER"
echo "  Seeds:            ${SEEDS[*]}"
echo "  Conditions:       3 (FA+drop, FP μ=1.0 γ-inexact, FP μ=0.01 γ-inexact)"
echo "  Total:            9 jobs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- seed $seed ---"

    # 1. FedAvg + drop-stragglers (Li 2020 §5.2 FedAvg arm at peak heterogeneity).
    submit_flower fedavg 0.0 "$seed" \
        "mn_storm_L4_fedavg_drop_s${seed}" \
        "--drop-stragglers"

    # 2. ⭐ FedProx μ=1.0 + γ-inexact (Li 2020 §5.2 FedProx arm; μ=1 was the
    #    best value on Li 2020's most-heterogeneous settings, §5.3.2).
    submit_flower fedprox 1.0 "$seed" \
        "mn_storm_L4_fedprox_mu1.0_s${seed}" \
        ""

    # 3. FedProx μ=0.01 ablation (tests sensitivity to μ choice in this regime;
    #    matches the value used elsewhere in the thesis).
    submit_flower fedprox 0.01 "$seed" \
        "mn_storm_L4_fedprox_mu0.01_s${seed}" \
        ""
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
echo "Expected result (predicted from Li 2020 §5.3.2 + NIID-Bench Table III):"
echo "  Condition 1 (FedAvg + drop):      macro-F1 ≈ 0.25-0.35 (rare-class blind)"
echo "  Condition 2 (FedProx μ=1.0):       macro-F1 ≈ 0.45-0.55 (predicted winner)"
echo "  Condition 3 (FedProx μ=0.01):      macro-F1 between (1) and (2)"
echo "  → Headline gap (1 → 2) should be 0.10-0.20 macro-F1"
echo ""
echo "After all 9 jobs finish, on the Mac run:"
echo "  python fl_dermamnist/analysis/analyse_fedprox_perfect_storm_L4.py"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
