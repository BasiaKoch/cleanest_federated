#!/bin/bash
# Experiment B — Asymmetric per-client μ on L4.
#
# Motivation. The single-seed L4 finding shows the entire FedProx-vs-FedAvg
# macro-F1 deficit (Δ = -0.026) is concentrated in ONE class (vascular
# lesions: 0.702 FedAvg → 0.481 FedProx, Δ = -0.221). FedProx is better
# or equal on 6 of 7 classes; it just suppresses the rare class that the
# small client (Client 1, 10% of data) holds exclusively.
#
# Hypothesis. The uniform μ = 0.01 over-regularises Client 1 — whose
# update direction is most divergent from the global anchor BECAUSE
# Client 1 is the specialist site. Setting μ_1 = 0 (no proximal anchor
# on the small specialist) while keeping μ_0 = 0.01 (anchor the
# dominant client) should recover vascular-class signal without
# sacrificing common-class wins.
#
# Literature positioning. This is NOT novel — it's an ablation of:
#   - Yao et al. 2024 (NeurIPS, arXiv:2410.08934, "Effect of
#     Personalization in FedProx") — proves the OPTIMAL μ depends on
#     per-client heterogeneity. Theoretical justification for per-client μ.
#   - HAPI-FedProx (Springer 2024, DOI:10.1007/978-3-032-11733-5_17)
#     — adapts μ per client via a local-vs-global heterogeneity index.
#     Closest direct precedent.
#   - FedPBS (arXiv:2603.13909) — selectively applies the proximal
#     correction by client batch size (effectively per-client μ).
# DISTINCT from Ditto / pFedMe / APFL / FedAMP — those train a
# client-specific model; we vary only the local regularisation strength
# on a single shared global model.
#
# Theoretical regime. Li et al. 2020 (arXiv:1812.06127) Theorem 4
# assumes uniform μ across clients. Setting μ_1 = 0 puts the design
# OUTSIDE Li 2020's proved regime but INSIDE Yao 2024's per-client
# minimax framework.
#
# Design (4 conditions × 3 seeds = 12 jobs, ~12 GPU-h on L4):
#
#   Condition                          μ_0       μ_1     Hypothesis
#   ---------------------------------------------------------------------
#   1. FedAvg                          0.0       0.0     no proximal anchor
#   2. Symmetric FedProx               0.01      0.01    baseline (suspect: hurts vascular)
#   3. Asymmetric "anchor-large" ⭐    0.01      0.0     anchor dominant, free specialist
#   4. Asymmetric "anchor-small" CTL   0.0       0.01    REVERSE control — direction matters
#
# The 4-th condition is the critical control. Without it, condition 3
# vs condition 1 is confounded with "any reduction in average μ helps".
# With it, we can attribute any rare-class recovery to the DIRECTION
# of asymmetry (anchor-large is the specific Yao 2024 prediction).
#
# Set SKIP_CONTROL=1 to drop condition 4 and run 9 jobs only.
#
# Seeds. 42, 123, 456. The reverse-asymmetric arm doubles as a sanity
# check that the per-client μ plumbing is symmetric in cid-space.
#
# Outputs:
#   fl_dermamnist/results/asymmetric_mu_L4/
#       test_at_best_{algo}_mu{...}{_muPC-c0m...-c1m...}_E20_s{42,123,456}.json
#
# Usage
# -----
#   bash fl_dermamnist/scripts/submit_asymmetric_mu_L4.sh                   # submit 12 jobs
#   SKIP_CONTROL=1 bash fl_dermamnist/scripts/submit_asymmetric_mu_L4.sh    # 9 jobs (drop ctrl)
#   DRY_RUN=1 bash fl_dermamnist/scripts/submit_asymmetric_mu_L4.sh         # print only
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

E_MAX=20
NUM_ROUNDS=150
MU_PROX=0.01
PARTITION=two_client_90_10_rare_stress

SEEDS=(42 123 456)

DRY_RUN="${DRY_RUN:-0}"
SKIP_CONTROL="${SKIP_CONTROL:-0}"

FLOWER_TPL="$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh"
OUTDIR="fl_dermamnist/results/asymmetric_mu_L4"
mkdir -p "$REPO_ROOT/$OUTDIR"

JOBS_SUMMARY=()
FAILED=()

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

n_conditions=4
if [ "$SKIP_CONTROL" = "1" ]; then
    n_conditions=3
fi
total_jobs=$(( ${#SEEDS[@]} * n_conditions ))

echo "============================================================"
echo "Experiment B — Asymmetric per-client μ on L4 (variance isolation)"
echo "  Partition: $PARTITION"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Conditions: $n_conditions"
echo "  Total:     $total_jobs jobs"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- seed $seed ---"

    # 1. FedAvg baseline.
    submit_flower fedavg 0.0 "$seed" "mn_asym_L4_fedavg_s${seed}" ""

    # 2. Symmetric FedProx baseline.
    submit_flower fedprox "$MU_PROX" "$seed" "mn_asym_L4_fedproxSym_s${seed}" ""

    # 3. ⭐ Asymmetric anchor-large (μ_0=0.01, μ_1=0.0).
    submit_flower fedprox "$MU_PROX" "$seed" \
        "mn_asym_L4_fedproxAnchorLarge_s${seed}" \
        "--mu-per-client 0:$MU_PROX,1:0.0"

    # 4. Asymmetric anchor-small CONTROL (μ_0=0.0, μ_1=0.01).
    if [ "$SKIP_CONTROL" != "1" ]; then
        submit_flower fedprox "$MU_PROX" "$seed" \
            "mn_asym_L4_fedproxAnchorSmall_s${seed}" \
            "--mu-per-client 0:0.0,1:$MU_PROX"
    fi
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
echo "Decision rule after all jobs finish:"
echo "  - If 'anchor-large' (cond 3) recovers vascular F1 above the"
echo "    symmetric-FedProx baseline AND outperforms 'anchor-small' (cond 4)"
echo "    by more than 3-seed SD → Yao 2024 prediction empirically supported"
echo "  - If 3 and 4 are statistically indistinguishable → the direction"
echo "    of asymmetry does not matter on this task (negative finding)"
echo ""
echo "Analysis script:"
echo "  python fl_dermamnist/analysis/analyse_asymmetric_mu_L4.py"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
