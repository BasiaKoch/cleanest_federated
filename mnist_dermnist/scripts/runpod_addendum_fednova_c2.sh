#!/bin/bash
# Runpod addendum — 3 missing/broken FedNova C2 jobs to complete the
# system_het_random_fednova sweep to 10 paired seeds.
#
# What's added:
#   - fednova s123     (was MISSING entirely)
#   - fednova s789     (was MISSING entirely)
#   - fednova s8675309 (was the numpy-overflow bug; now re-runs with fix)
#
# Why this is being run after Issue 2 was diagnosed:
#   The Issue 2 finding (FedNova catastrophically fails under random stragglers
#   + class imbalance) is based on 7 existing data points (all valid-but-low
#   macro_f1 in [0.114, 0.296]). Filling the dataset to 10 seeds:
#     (a) Strengthens the Issue 2 thesis claim if all 3 also collapse
#         (10/10 vs 7/8 is materially stronger statistical evidence).
#     (b) Provides a potential counter-example: s8675309 under the seed
#         fix MIGHT recover, giving direct evidence that the previous
#         collapse was not purely algorithmic.
#     (c) s123 and s789 must be run regardless — they were never produced.
#
# Run AFTER `runpod_resubmit_missing.sh` AND `runpod_addendum_provenance.sh`
# have both completed, so this isn't a third concurrent Flower simulation
# on the same GPU.
#
# Wall-clock: ~50 minutes (3 jobs × ~16 min on RTX 4090 solo, or 25 min
# each if it ends up running concurrently with something else).
#
# Usage on the pod:
#   cd /workspace/cleanest_federated
#   git pull origin main
#   tmux new -s fednovaC2 -d "bash mnist_dermnist/scripts/runpod_addendum_fednova_c2.sh 2>&1 | tee /workspace/runpod_addendum_fednova_c2_log.txt"
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUT="mnist_dermnist/results/system_het_random_fednova"
mkdir -p "$OUT"

echo "============================================================"
echo " FedNova C2 addendum (3 missing/broken jobs)"
echo " started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " host=$(hostname)  torch=$(python -c 'import torch; print(torch.__version__)')"
echo "============================================================"
echo ""

# Block 0: delete the broken s8675309 JSON so the idempotent runner re-fires it
echo "==== Block 0: delete known-bad s8675309 FedNova JSON ===="
BAD="$OUT/test_at_best_fednova_mu0.0_E20_sh-random_stragglers_s8675309.json"
if [ -f "$BAD" ]; then
    macro=$(python -c "import json; print(json.load(open('$BAD'))['macro_f1'])" 2>/dev/null || echo "-1")
    is_bad=$(python -c "print(1 if float($macro) < 0.20 else 0)" 2>/dev/null || echo "0")
    if [ "$is_bad" = "1" ]; then
        echo "  DELETE  $BAD  (macro_f1=$macro)"
        stem="fednova_mu0.0_E20_sh-random_stragglers_s8675309"
        rm -f "$BAD" \
              "$OUT/history_${stem}.csv" \
              "$OUT/test_predictions_${stem}.npz" \
              "$OUT/client_update_norms_${stem}.csv"
    else
        echo "  KEEP    $BAD  (macro_f1=$macro — already real training)"
    fi
else
    echo "  GONE    $BAD  (already absent)"
fi
echo ""

run_fednova_c2() {
    local seed="$1"
    local stem="fednova_mu0.0_E20_sh-random_stragglers_s${seed}"
    if [ -f "$OUT/test_at_best_${stem}.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $stem (already on disk)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $stem"
    PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_fednova_flower \
        --seed "$seed" \
        --local-epochs 20 --num-rounds 150 \
        --partition balanced_paired_7_clients --device cuda \
        --npz-path dermamnist_64.npz --out-dir "$OUT" \
        --system-het-mode random_stragglers \
        --straggler-fraction 0.5 \
        --log-update-norms \
        && echo "[$(date +%H:%M:%S)] DONE  $stem" \
        || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
}

echo "==== Block 1: 3 FedNova C2 jobs (s123, s789, s8675309) ===="
run_fednova_c2 123
run_fednova_c2 789
run_fednova_c2 8675309

echo ""
echo "============================================================"
echo " FedNova C2 addendum complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " To verify:"
echo "   ls $OUT/test_at_best_*.json | wc -l   # expect 10"
echo ""
echo " To see the per-seed macro_f1 (Issue 2 confirmation/refutation):"
echo "   python -c \"import json, glob; [print(f, json.load(open(f))['macro_f1']) for f in sorted(glob.glob('$OUT/test_at_best_*.json'))]\""
echo "============================================================"
