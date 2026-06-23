#!/bin/bash
# Morning batch submission via pure-PyTorch path (Ray-free).
#
# Submits all 4 experiments that are portable to run_one.py without
# code changes. The pure-PyTorch runner does NOT depend on Ray/Flower,
# so it bypasses the GCS startup issue entirely. The smoke-test job
# 29906694 (gpu-q-32) confirmed this path works on compute nodes.
#
# Submits 30 jobs total:
#   - node-pinned L4         : 6 jobs (3 seeds × 2 algos)
#   - extended-rounds L3     : 6 jobs (3 seeds × 2 algos, num_rounds=250)
#   - μ-sweep ladder         : 12 jobs (3 levels × 4 μ values, seed 42)
#   - 90/10 baseline full    : 6 jobs (3 seeds × 2 algos on L4)
#
# Three experiments are NOT submitted here because they need CLI flags
# only present in run_one_flower.py (--drop-stragglers, --mu-per-client):
#   - fedprox_perfect_storm_L4  (needs --drop-stragglers)
#   - li2020_asymmetric_L4       (needs --drop-stragglers)
#   - asymmetric_mu_L4           (needs --mu-per-client)
# These can be added to run_one.py separately and submitted after.
#
# FedNova has its own Flower-only runner; not currently portable.
#
# Usage
# -----
#   bash infra/slurm/morning_submit_pytorch.sh
#   DRY_RUN=1 bash infra/slurm/morning_submit_pytorch.sh
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

PYTORCH_TPL="$REPO_ROOT/infra/slurm/slurm_template_pytorch.sh"
DRY_RUN="${DRY_RUN:-0}"

E_MAX=20
MU_PROX=0.01

JOBS=()
FAILED=()

submit() {
    local algo="$1" mu="$2" seed="$3" rounds="$4" partition="$5" out_dir="$6" jobname="$7" extra="$8"
    mkdir -p "$REPO_ROOT/$out_dir"
    local args=("$algo" "$mu" "$seed" "$E_MAX" "$out_dir" "$partition" "--num-rounds $rounds --log-update-norms $extra")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY-RUN: sbatch --job-name=$jobname $PYTORCH_TPL ${args[*]}"
        return
    fi
    local jid
    if ! jid=$(sbatch --parsable --job-name="$jobname" "$PYTORCH_TPL" "${args[@]}"); then
        echo "  FAILED to submit: $jobname" >&2
        FAILED+=("$jobname")
        return 1
    fi
    JOBS+=("$jid  $jobname")
    echo "  $jid  $jobname"
    sleep 2  # space submissions out
}

echo "============================================================"
echo "MORNING SUBMIT — pure-PyTorch path (Ray-free)"
echo "============================================================"

# --- 1. NODE-PINNED L4 (3 seeds × 2 algos = 6 jobs) ---
echo ""
echo "[1/4] Node-pinned L4 — variance isolation"
for seed in 42 123 456; do
    submit fedavg  0.0       "$seed" 150 \
        "two_client_90_10_rare_stress" \
        "fl_dermamnist/results/node_pinned_L4" \
        "mn_pin_L4_fedavg_s${seed}" ""
    submit fedprox "$MU_PROX" "$seed" 150 \
        "two_client_90_10_rare_stress" \
        "fl_dermamnist/results/node_pinned_L4" \
        "mn_pin_L4_fedprox_s${seed}" ""
done

# --- 2. EXTENDED-ROUNDS L3 (3 seeds × 2 algos = 6 jobs, 250 rounds) ---
echo ""
echo "[2/4] Extended-rounds L3 — convergence fix"
for seed in 42 123 456; do
    submit fedavg  0.0       "$seed" 250 \
        "two_client_70_30_rare_enriched" \
        "fl_dermamnist/results/extended_rounds_L3" \
        "mn_ext_L3_fedavg_s${seed}" ""
    submit fedprox "$MU_PROX" "$seed" 250 \
        "two_client_70_30_rare_enriched" \
        "fl_dermamnist/results/extended_rounds_L3" \
        "mn_ext_L3_fedprox_s${seed}" ""
done

# --- 3. μ-SWEEP LADDER (3 levels × 4 μ values, seed 42 = 12 jobs) ---
echo ""
echo "[3/4] μ-sweep across heterogeneity ladder"
declare -A LEVEL_PARTITIONS=(
    ["L0"]="two_client_50_50_stratified_iid"
    ["L2"]="two_client_50_50_label_skew_only"
    ["L4"]="two_client_90_10_rare_stress"
)
for level in L0 L2 L4; do
    partition="${LEVEL_PARTITIONS[$level]}"
    out_dir="fl_dermamnist/results/mu_sweep_ladder/${level}_${partition}"
    for mu in 0.001 0.01 0.1 1.0; do
        submit fedprox "$mu" 42 150 \
            "$partition" "$out_dir" \
            "mn_muswp_${level}_mu${mu}_s42" ""
    done
done

# --- 4. 90/10 BASELINE FULL (3 seeds × 2 algos = 6 jobs) ---
echo ""
echo "[4/4] 90/10 baseline — full 3-seed promotion"
for seed in 42 123 456; do
    submit fedavg  0.0       "$seed" 150 \
        "two_client_90_10_rare_stress" \
        "fl_dermamnist/results/two_client_90_10_rare_stress" \
        "mn_base_2c9010_fedavg_s${seed}" ""
    submit fedprox "$MU_PROX" "$seed" 150 \
        "two_client_90_10_rare_stress" \
        "fl_dermamnist/results/two_client_90_10_rare_stress" \
        "mn_base_2c9010_fedprox_s${seed}" ""
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete. Set DRY_RUN=0 to actually submit."
else
    echo "Submitted ${#JOBS[@]} jobs (${#FAILED[@]} failed)."
fi
echo "============================================================"
echo ""
echo "Monitor:    squeue -u \$USER --format='%.10i %.45j %.8T %.10M %.20R'"
echo "Status:     sacct -u \$USER --starttime now-1hour --format=State,Elapsed,JobName%30 --noheader | head"
echo ""
echo "Once jobs complete, run analyses:"
echo "  python fl_dermamnist/analysis/analyse_node_pinned_L4.py"
echo "  python fl_dermamnist/analysis/analyse_extended_rounds_L3.py"
echo "  python fl_dermamnist/analysis/analyse_mu_sweep_ladder.py"
echo ""
echo "Not submitted (need CLI flags only in run_one_flower.py):"
echo "  - perfect_storm_L4    (needs --drop-stragglers)"
echo "  - li2020_asymmetric   (needs --drop-stragglers)"
echo "  - asymmetric_mu_L4    (needs --mu-per-client)"
echo "  - fednova_unequal_E   (FedNova has own Flower-only runner)"
echo ""
echo "To enable these via pure-PyTorch, add the flags to run_one.py first."

if [ ${#FAILED[@]} -ne 0 ]; then
    echo ""
    echo "WARNING: ${#FAILED[@]} submission(s) failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
