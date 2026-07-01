#!/bin/bash
# Runpod sequential runner - finalises the missing/corrupted jobs across
# the Flower-runtime sweeps after the seeding.py fix landed.
#
# Idempotent: each job is skipped if its test_at_best_*.json already exists
# AND is not on the known-bad-s8675309 cleanup list. Safe to re-run after
# Ctrl-C, after a pod crash, or after partial completion.
#
# What this script does:
#   Block 0: deletes 7 known-bad s8675309 JSONs (and their history/npz
#            companions) that were produced by the numpy-seed-overflow
#            bug fixed in fl/seeding.py. The runner will then re-run them.
#   Block 1: 6 missing flower_C0_baseline jobs.
#   Block 2: 5 missing system_het_fixed (C1) jobs + 1 bad-s8675309 rerun.
#   Block 3: 5 missing system_het_random (C2) jobs + 1 bad-s8675309 rerun.
#   Block 4: 5 bad-s8675309 reruns across iid/, dirichlet_a01/,
#            specialist_partition/.
#
# Intentionally OMITTED:
#   - FedNova C2 (system_het_random_fednova): we have evidence that
#     FedNova under random stragglers with class-imbalanced data
#     amplifies E=1-2 stragglers by ~9.27x and collapses the global
#     model to majority-class prediction. 7/8 existing seeds confirm
#     this failure mode (macro_f1 in [0.114, 0.296]); re-running with
#     the seeding fix will not change this. Documented as a thesis
#     finding rather than re-run.
#
# Usage on the pod:
#   cd /workspace/cleanest_federated
#   git pull origin main      # gets fl/seeding.py + fixed clients
#   bash infra/runpod/runpod_resubmit_missing.sh 2>&1 | tee /workspace/runpod_log.txt
#
# Wall-clock estimate on RTX 4090: ~15h sequential (20 jobs * ~45 min).
# After completion, rsync results back to the laptop.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p fl_dermamnist/results/flower_C0_baseline \
         fl_dermamnist/results/system_het_fixed \
         fl_dermamnist/results/system_het_random \
         fl_dermamnist/results/iid \
         fl_dermamnist/results/dirichlet_a01 \
         fl_dermamnist/results/specialist_partition

# === Block 0: delete known-bad s8675309 JSONs so the runner re-runs them ===
echo "============================================================"
echo " Runpod resubmission run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " host=$(hostname)  torch=$(python -c 'import torch; print(torch.__version__)')"
echo " flwr=$(python -c 'import flwr; print(flwr.__version__)')"
echo "============================================================"
echo ""
echo "==== Block 0 : cleanup of known-bad s8675309 outputs ===="
BAD_LIST=(
    fl_dermamnist/results/iid/test_at_best_fedavg_mu0.0_E20_s8675309.json
    fl_dermamnist/results/iid/test_at_best_fedprox_mu0.01_E20_s8675309.json
    fl_dermamnist/results/dirichlet_a01/test_at_best_fedprox_mu0.01_E20_s8675309.json
    fl_dermamnist/results/specialist_partition/test_at_best_fedavg_mu0.0_E20_s8675309.json
    fl_dermamnist/results/specialist_partition/test_at_best_fedprox_mu0.01_E20_s8675309.json
    fl_dermamnist/results/system_het_fixed/test_at_best_fedprox_mu0.01_E20_sh-fixed_stragglers_s8675309.json
    fl_dermamnist/results/system_het_random/test_at_best_fedavg_mu0.0_E20_sh-random_stragglers_s8675309.json
)
for f in "${BAD_LIST[@]}"; do
    if [ -f "$f" ]; then
        # Verify the file truly is bad (macro_f1 < 0.20) before deleting
        macro=$(python -c "import json; print(json.load(open('$f')).get('macro_f1', -1))" 2>/dev/null || echo "-1")
        is_bad=$(python -c "print(1 if float($macro) < 0.20 else 0)" 2>/dev/null || echo "0")
        if [ "$is_bad" = "1" ]; then
            echo "  DELETE  $f  (macro_f1=$macro)"
            rm -f "$f"
            # Also remove the companion history CSV + predictions npz
            stem=$(basename "$f" .json | sed 's/^test_at_best_//')
            outdir=$(dirname "$f")
            rm -f "$outdir/history_${stem}.csv" "$outdir/test_predictions_${stem}.npz" \
                  "$outdir/client_update_norms_${stem}.csv"
        else
            echo "  KEEP    $f  (macro_f1=$macro — not bad)"
        fi
    else
        echo "  GONE    $f  (already absent)"
    fi
done

run_flower() {
    local algo="$1" mu="$2" seed="$3" outdir="$4" partition="$5" sh_mode="$6" extra="$7"
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
        PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
            --algorithm "$algo" --mu "$mu" --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition "$partition" --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    else
        PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
            --algorithm "$algo" --mu "$mu" --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition "$partition" --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" \
            --system-het-mode "$sh_mode" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    fi
}

run_fednova() {
    local seed="$1" outdir="$2" partition="$3" sh_mode="$4" extra="$5"
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
        PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_fednova_flower \
            --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition "$partition" --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    else
        PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_fednova_flower \
            --seed "$seed" \
            --local-epochs 20 --num-rounds 150 \
            --partition "$partition" --device cuda \
            --npz-path dermamnist_64.npz --out-dir "$outdir" \
            --system-het-mode "$sh_mode" $extra \
            && echo "[$(date +%H:%M:%S)] DONE  $stem" \
            || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
    fi
}

# === Block 1: 6 missing flower_C0_baseline jobs ===
echo ""
echo "==== Block 1/4 : flower_C0_baseline (6 jobs, ~4.5h) ===="
run_flower fedavg  0.0  8675309 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform ""
run_flower fedprox 0.01 8675309 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform ""
run_flower fedprox 0.01 161803  fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform ""
run_fednova            2024     fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform ""
run_fednova            8675309  fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform ""
run_fednova            161803   fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform ""

# === Block 2: 5 missing + 1 bad-s8675309 rerun, system_het_fixed C1 ===
echo ""
echo "==== Block 2/4 : system_het_fixed C1 (6 jobs, ~4.5h) ===="
EXTRA_C1="--straggler-epochs 5 --fixed-straggler-ids 5,6 --log-update-norms"
run_flower fedprox 0.01 42      fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
run_flower fedprox 0.01 999     fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
run_flower fedprox 0.01 2024    fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
run_flower fedavg  0.0  8675309 fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
run_flower fedavg  0.0  161803  fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
run_flower fedprox 0.01 8675309 fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"

# === Block 3: 5 missing + 1 bad-s8675309 rerun, system_het_random C2 ===
echo ""
echo "==== Block 3/4 : system_het_random C2 (6 jobs, ~4.5h) ===="
EXTRA_C2="--straggler-fraction 0.5 --log-update-norms"
run_flower fedavg  0.0  123     fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
run_flower fedprox 0.01 789     fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
run_flower fedavg  0.0  2024    fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
run_flower fedprox 0.01 8675309 fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
run_flower fedprox 0.01 161803  fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
run_flower fedavg  0.0  8675309 fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"

# === Block 4: 5 bad-s8675309 reruns across iid, dirichlet, specialist ===
echo ""
echo "==== Block 4/4 : statistical-het bad-s8675309 reruns (5 jobs, ~3.8h) ===="
# iid (2 jobs)
run_flower fedavg  0.0  8675309 fl_dermamnist/results/iid                  iid_7_clients               uniform ""
run_flower fedprox 0.01 8675309 fl_dermamnist/results/iid                  iid_7_clients               uniform ""
# dirichlet_alpha01 (1 job - fedavg s8675309 is already valid on disk)
run_flower fedprox 0.01 8675309 fl_dermamnist/results/dirichlet_a01        dirichlet_alpha01_7_clients uniform ""
# specialist (2 jobs)
run_flower fedavg  0.0  8675309 fl_dermamnist/results/specialist_partition specialist_7_clients        uniform ""
run_flower fedprox 0.01 8675309 fl_dermamnist/results/specialist_partition specialist_7_clients        uniform ""

# === FedNova C2 INTENTIONALLY OMITTED: see header comment ===

echo ""
echo "============================================================"
echo " All 23 jobs attempted at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " To verify completion:"
echo "   ls fl_dermamnist/results/flower_C0_baseline/test_at_best_*.json     | wc -l   # expect 30"
echo "   ls fl_dermamnist/results/system_het_fixed/test_at_best_*.json       | wc -l   # expect 20"
echo "   ls fl_dermamnist/results/system_het_random/test_at_best_*.json      | wc -l   # expect 20"
echo "   ls fl_dermamnist/results/iid/test_at_best_*.json                    | wc -l   # expect 20"
echo "   ls fl_dermamnist/results/dirichlet_a01/test_at_best_*.json          | wc -l   # expect 20"
echo "   ls fl_dermamnist/results/specialist_partition/test_at_best_*.json   | wc -l   # expect 20"
echo "   ls fl_dermamnist/results/system_het_random_fednova/test_at_best_*.json | wc -l  # expect 8 (FedNova C2 documented as failure mode)"
echo "============================================================"
