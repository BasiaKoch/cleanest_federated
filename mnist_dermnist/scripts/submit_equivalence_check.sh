#!/bin/bash
# Submit the full-scale equivalence-verification sweep:
#
#   2 seeds × 2 algorithms × full config (E=20, R=150) via the Flower runtime
#   = 4 SLURM jobs at ~1 GPU-hour each.
#
# Seeds 42 and 8675309 were chosen a priori because they are the smallest-
# and largest-Δ contributors to the existing 10-seed headline mean (Δ = +0.0685
# and +0.0953 respectively), so any framework-induced divergence is most
# detectable at these endpoints.
#
# The Flower-produced test macro-F1 values for these 4 runs are then
# compared to the existing pure-PyTorch results in
#   mnist_dermnist/results/headline/test_at_best_<algo>_mu*_E20_s{42,8675309}.json
# via
#   PYTHONPATH=. python -m mnist_dermnist.experiments.compare_equivalence_full_scale
#
# Acceptance criterion: |Δ macro-F1| ≤ 0.02 between runtimes on each of the
# 4 paired runs. Any larger discrepancy indicates a runtime bug; smaller
# than that is consistent with the CUDA/RNG noise floor already documented
# in the methodology, and the existing 10-seed headline data is preserved.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS_FOR_VERIFY=(42 8675309)
MU=0.01
PARTITION=balanced_paired_7_clients
OUT_DIR=mnist_dermnist/results/headline_flower_verify

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/mnist_dermnist/logs"

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_verify_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed"
    FAILED+=("$algo $mu $seed")
  fi
  sleep 3
}

for s in "${SEEDS_FOR_VERIFY[@]}"; do
  submit fedavg  0.0  "$s"
  submit fedprox $MU  "$s"
done

echo ""
echo "Submitted equivalence-verification sweep:"
echo "  - 2 seeds × 2 algorithms = 4 jobs at full E=$LOCAL_EPOCHS R=150 via Flower"
echo "  - Output: $OUT_DIR"
echo "Once all 4 jobs complete, compare against existing pure-PyTorch results:"
echo "  PYTHONPATH=. python -m mnist_dermnist.experiments.compare_equivalence_full_scale"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
