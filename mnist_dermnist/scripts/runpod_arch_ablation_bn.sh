#!/bin/bash
# Architecture ablation: DermMNISTCNN with BatchNorm (vs the headline
# GroupNorm variant).
#
# Hypothesis: FedProx's parameter-side proximal anchor partially compensates
# for the well-documented BN-running-stats × non-IID interaction (Li et al.
# 2021, "FedBN"). Predicted outcome: Δ_FedProx-FedAvg on the BN variant is
# LARGER than the GN headline (~+0.007 on Flower, ~+0.027 on pure-PyTorch).
#
# Design:
#   - 3 paired seeds (42, 123, 456) to match the centralised baseline seeds
#     and the first 3 entries of the headline paired-seed protocol.
#   - 2 algorithms (FedAvg, FedProx μ=0.01).
#   - 1 partition (balanced_paired_7_clients, the engineered headline).
#   - Same hyperparameters as the headline (E=20, R=150, lr=0.01, m=0.9, B=32).
#   - All runs use --model-variant bn so the runner instantiates
#     DermMNISTCNN_BN (BatchNorm2d in place of GroupNorm).
#   - Outputs saved to mnist_dermnist/results/arch_ablation_bn/ — a separate
#     directory from the headline so the two variants cannot be mixed in
#     downstream analysis. Filenames additionally carry an '_arch-bn' tag
#     so a stray file can always be identified as BN-variant.
#
# Total: 6 jobs, ~1.5h wall-clock on RTX 4090 (BN is ~5% slower than GN).
#
# Run only AFTER the main runner (runpod_resubmit_missing.sh) has finished
# so this does not compete for GPU.
#
# Usage on the pod:
#   cd /workspace/cleanest_federated
#   git pull origin main
#   tmux new -s archbn -d "bash mnist_dermnist/scripts/runpod_arch_ablation_bn.sh 2>&1 | tee /workspace/runpod_arch_ablation_bn_log.txt"
#
# Compare to the headline GN seeds with:
#   python mnist_dermnist/scripts/check_arch_ablation.py
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUT="mnist_dermnist/results/arch_ablation_bn"
mkdir -p "$OUT"

echo "============================================================"
echo " Architecture ablation (BatchNorm variant) — run started at"
echo "   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " host=$(hostname)  torch=$(python -c 'import torch; print(torch.__version__)')"
echo " variant: DermMNISTCNN_BN (BatchNorm2d, no GroupNorm)"
echo " partition: balanced_paired_7_clients"
echo " seeds: 42, 123, 456     (3 paired seeds)"
echo " algos: fedavg, fedprox  (2 algorithms)"
echo " expected: 6 jobs, ~1.5h on a 4090"
echo "============================================================"
echo ""

run_bn() {
    local algo="$1" mu="$2" seed="$3"
    local stem="${algo}_mu${mu}_E20_arch-bn_s${seed}"
    if [ -f "$OUT/test_at_best_${stem}.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP  $stem (already on disk)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $stem"
    PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \
        --algorithm "$algo" --mu "$mu" --seed "$seed" \
        --local-epochs 20 --num-rounds 150 \
        --partition balanced_paired_7_clients --device cuda \
        --npz-path dermamnist_64.npz --out-dir "$OUT" \
        --model-variant bn \
        --log-update-norms \
        && echo "[$(date +%H:%M:%S)] DONE  $stem" \
        || echo "[$(date +%H:%M:%S)] FAIL  $stem rc=$?"
}

# 3 seeds * 2 algos = 6 jobs, sequential
for SEED in 42 123 456; do
    run_bn fedavg  0.0  "$SEED"
    run_bn fedprox 0.01 "$SEED"
done

echo ""
echo "============================================================"
echo " Architecture ablation complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Verify:"
echo "   ls $OUT/test_at_best_*.json | wc -l   # expect 6"
echo " Compare to GN headline:"
echo "   python mnist_dermnist/scripts/check_arch_ablation.py"
echo "============================================================"
