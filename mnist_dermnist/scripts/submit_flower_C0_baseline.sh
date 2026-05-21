#!/bin/bash
# PRIMARY HEADLINE SWEEP — Flower runtime, FedAvg + FedProx + FedNova,
# balanced_paired_7_clients, 10 paired seeds.
#
# Why this is now the primary headline (2026-05-21):
# The thesis treats Flower as the canonical FL framework. The existing
# pure-PyTorch headline at mnist_dermnist/results/headline/ is retained
# as the cross-runtime validation reference (its 10 paired seeds are
# the baseline against which submit_equivalence_check.sh verifies
# Flower-vs-PyTorch equivalence at the two extreme-Δ seeds). The
# canonical numeric headline going forward — including the FedNova
# arm needed for the three-algorithm comparison — is produced by this
# script and lives in mnist_dermnist/results/flower_C0_baseline/.
#
# Hyperparameters match the legacy headline exactly:
#   • Partition: balanced_paired_7_clients
#   • Algorithms: FedAvg (μ=0.0), FedProx (μ=0.01), FedNova (m=0.9)
#   • Seeds: 42 123 456 789 999 2024 31337 8675309 161803 271828
#   • R = 150 rounds, E = 20 local epochs, batch = 32, lr = 0.01
#   • Full participation (C = 1.0), uniform local-epoch schedule (no stragglers)
#   • --log-update-norms enabled so client drift is measured on the
#     headline (supports the "FedProx mitigates drift" claim with
#     direct evidence instead of inference).
#
# Compute: 30 jobs (10 seeds × 3 algorithms), ~30 GPU-hours on A100.
#
# This sweep also serves as the C0 baseline for system-heterogeneity H2
# inference (analyse_system_het.py refuses to run without it).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
PARTITION=balanced_paired_7_clients
OUT_DIR=mnist_dermnist/results/flower_C0_baseline

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/mnist_dermnist/logs"

# Drift logging is enabled on the headline so the thesis can substantiate
# the "FedProx mitigates client drift" claim with direct measurements
# (||w_k^{t+1} - w^t||_2 per round, per client) rather than inferring
# from per-class behaviour.
EXTRA_ARGS="--log-update-norms"

FAILED=()
submit_flower() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_C0_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" "$EXTRA_ARGS"; then
    echo "  FAILED: $algo mu=$mu seed=$seed"
    FAILED+=("$algo $mu $seed")
  fi
  sleep 3
}

submit_fednova() {
  local seed="$1"
  if ! sbatch \
    --job-name="mn_C0_fednova_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_fednova.sh" \
    "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" uniform "$EXTRA_ARGS"; then
    echo "  FAILED: fednova seed=$seed"
    FAILED+=("fednova $seed")
  fi
  sleep 3
}

for s in "${SEEDS[@]}"; do
  submit_flower fedavg  0.0  "$s"
  submit_flower fedprox $MU  "$s"
  submit_fednova "$s"
done

echo ""
echo "Submitted Flower C0 baseline:"
echo "  - 10 seeds × {FedAvg, FedProx, FedNova} = 30 jobs → $OUT_DIR"
echo "  Total: ~30 GPU-hours."
echo ""
echo "Once complete, these are the C0 numbers to use in H2 system-het inference."
echo "Do NOT use the pure-PyTorch headline as C0 against Flower C1/C2; the"
echo "runtime difference would confound the system-het manipulation."

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
