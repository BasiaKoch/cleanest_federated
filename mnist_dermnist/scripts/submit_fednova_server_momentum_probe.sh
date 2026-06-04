#!/bin/bash
# Publishable remedy: FedNova + server momentum (β = 0.9) on the
# random-straggler L4 / balanced-paired-7-clients protocol.
#
# Reference: Hsu, Qi & Brown 2019, FedAvgM (arxiv.org/abs/1909.06335);
# Cheng et al., AAAI 2024, "On the Role of Server Momentum in FL".
# Implementation in mnist_dermnist/fl_flower/strategy_fednova.py mirrors
# flwr.server.strategy.fedavgm.FedAvgM exactly (first-round init m_1 = g_1,
# subsequent m_t = β·m_{t-1} + g_t, applied as w_{t+1} = w_t − m_t).
#
# Tests the hypothesis that low-pass-filtering the noisy per-round FedNova
# pseudo-gradient (1/τ amplification varies wildly between rounds) is
# sufficient to recover the collapsed seeds without altering the
# normaliser math.
#
# Output dir: mnist_dermnist/results/system_het_random_fednova_servmom/
# Compute: ~5 GPU-h on A100 (5 seeds × ~1 GPU-h each).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SERVER_MOMENTUM=0.9
SEEDS=(31337 271828 161803 42 123)
PARTITION=balanced_paired_7_clients

FAILED=()
submit() {
  local seed="$1" out="$2" sh_mode="$3" extra="$4"
  if ! sbatch \
    --job-name="mn_fn_smom_${sh_mode}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_fednova.sh" \
    "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$sh_mode" "$extra"; then
    echo "  FAILED to submit: seed=$seed sh=$sh_mode"
    FAILED+=("$seed $sh_mode")
  fi
  sleep 3
}

OUT=mnist_dermnist/results/system_het_random_fednova_servmom
mkdir -p "$REPO_ROOT/$OUT"
for s in "${SEEDS[@]}"; do
  # NOTE: slurm_template_fednova.sh hardcodes --batch-size 10 --momentum 0.0,
  # but the collapsed-seed baseline runs used batch=32, momentum=0.9 (see
  # mnist_dermnist/results/system_het_random_fednova/test_at_best_*.json).
  # Argparse uses the LAST occurrence, so the extras below override the
  # template defaults and match the baseline protocol exactly.
  submit "$s" "$OUT" random_stragglers \
    "--batch-size 32 --momentum 0.9 --straggler-fraction 0.5 --log-update-norms --server-momentum ${SERVER_MOMENTUM}"
done

echo ""
echo "Submitted FedNova server-momentum probe sweep:"
echo "  - random_stragglers × ${#SEEDS[@]} seeds (3 collapsed + 2 controls)"
echo "  - server_momentum = β = ${SERVER_MOMENTUM}  (Hsu 2019 default)"
echo "  - output: $OUT/"
echo ""
echo "Success criterion: macro-F1 > 0.4 for ALL three previously-collapsed"
echo "seeds confirms that round-to-round τ-jitter is the dominant failure mode."
echo "Healthy-control seeds (42, 123) may shift modestly — momentum is not"
echo "neutral on already-converging trajectories — but should remain > 0.3."
echo ""
echo "When complete:"
echo "  PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/peek_system_het_progress.py \\"
echo "      --dir mnist_dermnist/results/system_het_random_fednova_servmom"
echo "Monitor with:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
