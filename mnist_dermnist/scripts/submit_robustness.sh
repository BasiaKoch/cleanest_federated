#!/bin/bash
# Submit the robustness-check sweeps to SLURM:
#   - Dirichlet-alpha = 0.1 (standard severe non-IID, Hsu 2019) — 20 jobs
#   - IID falsification control (FedProx should not significantly differ from FedAvg) — 20 jobs
#
# Same 10 seeds, E=20, R=150 as the headline sweep. Total: ~40 jobs ~50 GPU-h.
# Runtime: Flower 1.x simulation framework (slurm_template_flower.sh).
# All robustness sweeps use the Flower runtime; equivalence to the
# original pure-PyTorch loop (which produced the headline 10-seed data)
# is verified by submit_equivalence_check.sh.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3" out="$4" part="$5"
  if ! sbatch \
    --job-name="mn_${algo}_${part}_mu${mu}_E${LOCAL_EPOCHS}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$out" "$part"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed part=$part"
    FAILED+=("$algo $mu $seed $part")
  fi
  sleep 3
}

# --- Dirichlet alpha=0.1 sweep ---
DIR_OUT=mnist_dermnist/results/dirichlet_a01
mkdir -p "$REPO_ROOT/$DIR_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0 "$s" "$DIR_OUT" dirichlet_alpha01_7_clients
  submit fedprox $MU  "$s" "$DIR_OUT" dirichlet_alpha01_7_clients
done

# --- IID falsification control ---
IID_OUT=mnist_dermnist/results/iid
mkdir -p "$REPO_ROOT/$IID_OUT"
for s in "${SEEDS[@]}"; do
  submit fedavg  0.0 "$s" "$IID_OUT" iid_7_clients
  submit fedprox $MU  "$s" "$IID_OUT" iid_7_clients
done

echo ""
echo "Submitted robustness sweep:"
echo "  - dirichlet_alpha01_7_clients × 10 seeds × 2 algos = 20 jobs → $DIR_OUT"
echo "  - iid_7_clients              × 10 seeds × 2 algos = 20 jobs → $IID_OUT"
echo "Monitor with:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
