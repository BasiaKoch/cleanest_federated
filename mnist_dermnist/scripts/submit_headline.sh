#!/bin/bash
# Submit the full headline sweep to SLURM:
#   10 seeds × {FedAvg, FedProx(μ=0.01)} = 20 jobs, E=20, 150 rounds.
#   Partition: balanced_paired_7_clients (every class held by ≥2 clients).
set -uo pipefail   # NOT -e: a single sbatch failure must not abort the whole sweep

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
OUT_DIR=mnist_dermnist/results/headline
PARTITION=balanced_paired_7_clients     # ← the FedProx-favourable design
MU=0.01                                  # ← from CPU sweep (replace if HPC μ-sweep picks something else)

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/mnist_dermnist/logs"

SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
LOCAL_EPOCHS=20

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_${algo}_mu${mu}_E${LOCAL_EPOCHS}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed (will need manual resubmit)"
    FAILED+=("$algo $mu $seed")
  fi
  sleep 3   # was 1 — gives slurmctld time to settle and avoids RPC timeouts
}

for s in "${SEEDS[@]}"; do
  submit fedavg  0.0  "$s"
  submit fedprox $MU  "$s"
done

echo ""
echo "Submitted headline sweep: 10 seeds × 2 algos = 20 jobs."
echo "  partition: $PARTITION"
echo "  μ:         $MU"
echo "Monitor with:  squeue -u \$USER"
echo "When done:     bash mnist_dermnist/scripts/check_results.sh"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  echo "Re-run those manually via:"
  echo "  sbatch mnist_dermnist/scripts/slurm_template.sh <algo> <mu> <seed> $LOCAL_EPOCHS $OUT_DIR $PARTITION"
fi
