#!/bin/bash
# Runpod sequential runner — re-fires the 18 jobs that failed (or stalled)
# on Cambridge HPC across today's submission rounds.
#
# Idempotent: each job is skipped if its test_at_best_*.json already
# exists in the target directory. This means you can re-run this script
# safely after Ctrl-C, after a pod crash, or after partial completion.
#
# Total wall-clock estimate on RTX 4090: ~14 hours sequential
# (most jobs at E=20 R=150 are ~45 min on a 4090). For ~3-hour
# turnaround, split this file into 3 parallel pods, each running ~6 jobs.
#
# Usage on the pod:
#   cd /workspace/cleanest_federated
#   git pull origin main
#   bash mnist_dermnist/scripts/runpod_resubmit_missing.sh 2>&1 | tee /workspace/runpod_log.txt
#
# After completion, rsync results back to the laptop (see
# results/README.md or the conversation log for the rsync command).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p mnist_dermnist/results/flower_C0_baseline \
         mnist_dermnist/results/system_het_fixed \
         mnist_dermnist/results/system_het_random \
         mnist_dermnist/results/system_het_random_fednova

run_flower() {
    local algo="$1" mu="$2" seed="$3" outdir="$4" sh_mode="$5" extra="$6"
    local stem
    if [ "$sh_mode" = "uniform" ]; then
        stem="${algo}_mu${mu}_E20_s${seed}"
    else
        stem="${algo}_mu${mu}_E20_sh-${sh_mode}_s${seed}"
    fi
    if [ -f "$outdir/test_at_best_${stem}.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $stem (already on disk)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $stem"
    if [ "$sh_mode" = "uniform" ]; then
        PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \
            --algorithm "$algo" --mu "$mu" --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition balanced_paired_7_clients --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    else
        PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \
            --algorithm "$algo" --mu "$mu" --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition balanced_paired_7_clients --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" \
            --system-het-mode "$sh_mode" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    fi
}

run_fednova() {
    local seed="$1" outdir="$2" sh_mode="$3" extra="$4"
    local stem
    if [ "$sh_mode" = "uniform" ]; then
        stem="fednova_mu0.0_E20_s${seed}"
    else
        stem="fednova_mu0.0_E20_sh-${sh_mode}_s${seed}"
    fi
    if [ -f "$outdir/test_at_best_${stem}.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $stem (already on disk)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $stem"
    if [ "$sh_mode" = "uniform" ]; then
        PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_fednova_flower \
            --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition balanced_paired_7_clients --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    else
        PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_fednova_flower \
            --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition balanced_paired_7_clients --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" \
            --system-het-mode "$sh_mode" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    fi
}

echo "============================================================"
echo " Runpod resubmission run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " host=$(hostname)  torch=$(python -c 'import torch; print(torch.__version__)')"
echo " flwr=$(python -c 'import flwr; print(flwr.__version__)')"
echo "============================================================"

# === Block 1: 6 missing flower_C0_baseline jobs (uniform compute) ===
echo ""
echo "==== Block 1/4 : flower_C0_baseline (6 jobs, ~4.5h) ===="
run_flower fedavg  0.0  8675309 mnist_dermnist/results/flower_C0_baseline uniform ""
run_flower fedprox 0.01 8675309 mnist_dermnist/results/flower_C0_baseline uniform ""
run_flower fedprox 0.01 161803  mnist_dermnist/results/flower_C0_baseline uniform ""
run_fednova            2024     mnist_dermnist/results/flower_C0_baseline uniform ""
run_fednova            8675309  mnist_dermnist/results/flower_C0_baseline uniform ""
run_fednova            161803   mnist_dermnist/results/flower_C0_baseline uniform ""

# === Block 2: 5 missing system_het_fixed C1 jobs ===
echo ""
echo "==== Block 2/4 : system_het_fixed C1 (5 jobs, ~3.8h) ===="
EXTRA_C1="--straggler-epochs 5 --fixed-straggler-ids 5,6 --log-update-norms"
run_flower fedprox 0.01 42      mnist_dermnist/results/system_het_fixed fixed_stragglers "$EXTRA_C1"
run_flower fedprox 0.01 999     mnist_dermnist/results/system_het_fixed fixed_stragglers "$EXTRA_C1"
run_flower fedprox 0.01 2024    mnist_dermnist/results/system_het_fixed fixed_stragglers "$EXTRA_C1"
run_flower fedavg  0.0  8675309 mnist_dermnist/results/system_het_fixed fixed_stragglers "$EXTRA_C1"
run_flower fedavg  0.0  161803  mnist_dermnist/results/system_het_fixed fixed_stragglers "$EXTRA_C1"

# === Block 3: 5 missing system_het_random C2 jobs ===
echo ""
echo "==== Block 3/4 : system_het_random C2 (5 jobs, ~3.8h) ===="
EXTRA_C2="--straggler-fraction 0.5 --log-update-norms"
run_flower fedavg  0.0  123     mnist_dermnist/results/system_het_random random_stragglers "$EXTRA_C2"
run_flower fedprox 0.01 789     mnist_dermnist/results/system_het_random random_stragglers "$EXTRA_C2"
run_flower fedavg  0.0  2024    mnist_dermnist/results/system_het_random random_stragglers "$EXTRA_C2"
run_flower fedprox 0.01 8675309 mnist_dermnist/results/system_het_random random_stragglers "$EXTRA_C2"
run_flower fedprox 0.01 161803  mnist_dermnist/results/system_het_random random_stragglers "$EXTRA_C2"

# === Block 4: 2 missing FedNova C2 jobs ===
echo ""
echo "==== Block 4/4 : FedNova C2 (2 jobs, ~1.5h) ===="
run_fednova 123 mnist_dermnist/results/system_het_random_fednova random_stragglers "$EXTRA_C2"
run_fednova 789 mnist_dermnist/results/system_het_random_fednova random_stragglers "$EXTRA_C2"

echo ""
echo "============================================================"
echo " All 18 jobs attempted at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " To verify completion:"
echo "   ls mnist_dermnist/results/flower_C0_baseline/test_at_best_*.json | wc -l   # expect 30"
echo "   ls mnist_dermnist/results/system_het_fixed/test_at_best_*.json   | wc -l   # expect 20"
echo "   ls mnist_dermnist/results/system_het_random/test_at_best_*.json  | wc -l   # expect 20"
echo "   ls mnist_dermnist/results/system_het_random_fednova/test_at_best_*.json | wc -l  # expect 10"
echo "============================================================"
