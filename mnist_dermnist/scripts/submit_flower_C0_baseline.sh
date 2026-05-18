#!/bin/bash
# Submit a Flower-runtime "C0" baseline sweep for the system-heterogeneity
# section.
#
# Why this exists
# ---------------
# The existing 10-seed headline data (mnist_dermnist/results/headline/) was
# produced by the PURE-PYTHON reference loop. The system-heterogeneity
# sweep (submit_system_het.sh) and the FedNova sweep
# (submit_fednova_system_het.sh) both run through Flower.
#
# H2 inference in the system-heterogeneity section compares
#     Δ_s^c - Δ_s^{C0}
# between system-het condition c (Flower) and baseline C0 (... which is what?).
# If C0 is taken from the pure-PyTorch headline, the H2 contrast partially
# reflects Flower-vs-PyTorch runtime differences, not just the system-het
# manipulation.
#
# To avoid this confound, this script produces a Flower-runtime C0 baseline
# at the same 10 paired seeds (FedAvg + FedProx + FedNova at uniform E=20,
# no stragglers). H2 inference for the system-het section is then done
# using these Flower-C0 numbers, not the pure-PyTorch headline.
#
# Compute: 30 jobs (10 seeds × 3 algorithms) ~30 GPU-hours on A100.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
PARTITION=balanced_paired_7_clients
OUT_DIR=mnist_dermnist/results/flower_C0_baseline

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/mnist_dermnist/logs"

FAILED=()
submit_flower() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_C0_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION"; then
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
    "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" uniform ""; then
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
