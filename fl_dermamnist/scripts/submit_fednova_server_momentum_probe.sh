#!/bin/bash
# Publishable remedy: FedNova + server momentum (β = 0.9) on the
# random-straggler L4 / balanced-paired-7-clients protocol.
#
# Reference: Hsu, Qi & Brown 2019, FedAvgM (arxiv.org/abs/1909.06335);
# Cheng et al., AAAI 2024, "On the Role of Server Momentum in FL".
# Implementation in fl_dermamnist/fl_flower/strategy_fednova.py mirrors
# flwr.server.strategy.fedavgm.FedAvgM exactly (first-round init m_1 = g_1,
# subsequent m_t = β·m_{t-1} + g_t, applied as w_{t+1} = w_t − m_t).
#
# Tests the hypothesis that low-pass-filtering the noisy per-round FedNova
# pseudo-gradient (1/τ amplification varies wildly between rounds) is
# sufficient to recover the collapsed seeds without altering the
# normaliser math.
#
# Output dir: fl_dermamnist/results/system_het_random_fednova_servmom/
# Compute: ~5 GPU-h on A100 (5 seeds × ~1 GPU-h each).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SERVER_MOMENTUM=0.9
SEEDS=(42 123 8675309 31337 271828)   # Stage-2 pilot set (plan §6)
PARTITION=balanced_paired_7_clients

FAILED=()
submit() {
  local seed="$1" out="$2" sh_mode="$3" extra="$4"
  if ! sbatch \
    --job-name="mn_fn_smom_${sh_mode}_s${seed}" \
    "$REPO_ROOT/fl_dermamnist/scripts/slurm_template_fednova.sh" \
    "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$sh_mode" "$extra"; then
    echo "  FAILED to submit: seed=$seed sh=$sh_mode"
    FAILED+=("$seed $sh_mode")
  fi
  sleep 3
}

OUT=fl_dermamnist/results/system_het_random_fednova_servmom
mkdir -p "$REPO_ROOT/$OUT"
for s in "${SEEDS[@]}"; do
  # NOTE: slurm_template_fednova.sh hardcodes --batch-size 10 --momentum 0.0,
  # but the collapsed-seed baseline runs used batch=32, momentum=0.9 (see
  # fl_dermamnist/results/system_het_random_fednova/test_at_best_*.json).
  # Argparse uses the LAST occurrence, so the extras below override the
  # template defaults and match the baseline protocol exactly.
  submit "$s" "$OUT" random_stragglers \
    "--batch-size 32 --momentum 0.9 --straggler-fraction 0.5 --log-update-norms --server-momentum ${SERVER_MOMENTUM}"
done

echo ""
echo "Submitted FedNova server-momentum probe sweep:"
echo "  - random_stragglers × ${#SEEDS[@]} seeds (pilot: 2 born-collapse, 2 transient, 1 best)"
echo "  - server_momentum = β = ${SERVER_MOMENTUM}  (Hsu 2019 default)"
echo "  - output: $OUT/"
echo ""
echo "Success criterion: server momentum lifts the pilot seeds out of"
echo "majority-class collapse (final-round macro-F1 toward the FedAvg band"
echo "~0.49), implicating round-to-round τ-jitter as the failure mode."
echo ""
echo "When complete:"
echo "  PYTHONPATH=. python fl_dermamnist/analysis/peek_system_het_progress.py \\"
echo "      --dir fl_dermamnist/results/system_het_random_fednova_servmom"
echo "Monitor with:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
