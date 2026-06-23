#!/bin/bash
# Runpod addendum — provenance re-runs for dirichlet_a01.
#
# A second-pass audit found 4 dirichlet_a01 JSONs with valid macro_f1
# values but inconsistent JSON provenance metadata (missing `framework`,
# `runner_script`, `loss_type`, and `predictions_file` fields). These
# were produced by an older runner before the provenance plumbing was
# added; their macro_f1 numbers are usable for the headline table but
# the lack of framework attribution is a reproducibility gap.
#
# This addendum re-runs them under the current (post-seeding-fix) runner
# so all 20 dirichlet_a01 entries share consistent provenance.
#
# Run AFTER the main `runpod_resubmit_missing.sh` has completed.
#
# Usage:
#   cd /workspace/cleanest_federated
#   bash fl_dermamnist/scripts/runpod_addendum_provenance.sh 2>&1 | tee /workspace/runpod_addendum_log.txt
#
# Wall-clock: ~3h (4 jobs * ~45 min).
#
# NOT INCLUDED (intentionally):
#   - FedNova C2 reruns (s123, s789, s8675309 in system_het_random_fednova):
#     Issue 2 diagnosis shows the failure is algorithmic (~9.27x straggler
#     amplification under E_min=1 + class imbalance), not the numpy seed
#     overflow. Re-running with the seeding fix will not change the
#     collapse. Documented as a thesis finding instead.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUT="fl_dermamnist/results/dirichlet_a01"
mkdir -p "$OUT"

echo "============================================================"
echo " Runpod provenance addendum started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " host=$(hostname)  torch=$(python -c 'import torch; print(torch.__version__)')"
echo "============================================================"
echo ""

# Force-delete the old provenance-broken JSONs (and companion files)
# so the idempotent runner re-fires them. macro_f1 of these is in the
# valid range (0.44-0.50) so the standard skip-if-exists check would
# otherwise leave them in place.
echo "==== Block 0: delete provenance-broken dirichlet_a01 JSONs ===="
PROVENANCE_BROKEN=(
    fedavg_mu0.0_E20_s123
    fedavg_mu0.0_E20_s2024
    fedavg_mu0.0_E20_s8675309
    fedprox_mu0.01_E20_s31337
)
for stem in "${PROVENANCE_BROKEN[@]}"; do
    f="$OUT/test_at_best_${stem}.json"
    if [ -f "$f" ]; then
        echo "  DELETE  $f"
        rm -f "$f"
        rm -f "$OUT/history_${stem}.csv" \
              "$OUT/test_predictions_${stem}.npz" \
              "$OUT/client_update_norms_${stem}.csv"
    else
        echo "  GONE    $f  (already absent)"
    fi
done
echo ""

run_dirichlet() {
    local algo="$1" mu="$2" seed="$3"
    local stem="${algo}_mu${mu}_E20_s${seed}"
    if [ -f "$OUT/test_at_best_${stem}.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $stem (already on disk)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $stem"
    PYTHONPATH=. python -m fl_dermamnist.experiments.run_one_flower \
        --algorithm "$algo" --mu "$mu" --seed "$seed" \
        --local-epochs 20 --num-rounds 150 \
        --partition dirichlet_alpha01_7_clients --device cuda \
        --npz-path dermamnist_64.npz --out-dir "$OUT" \
        && echo "[$(date +%H:%M:%S)] DONE  $stem" \
        || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
}

echo "==== Block 1: dirichlet_a01 provenance re-runs (4 jobs, ~3h) ===="
run_dirichlet fedavg  0.0  123
run_dirichlet fedavg  0.0  2024
run_dirichlet fedavg  0.0  8675309
run_dirichlet fedprox 0.01 31337

echo ""
echo "============================================================"
echo " Provenance addendum complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Final check:"
echo "   ls $OUT/test_at_best_*.json | wc -l   # expect 20"
echo "============================================================"
