#!/bin/bash
# Submit the full headline sweep to SLURM (Flower runtime).
#   10 seeds × {FedAvg, FedProx(μ=0.01)} = 20 jobs, E=20, 150 rounds.
#   Partition: balanced_paired_7_clients (every class held by ≥2 clients).
#
# Runtime note (2026-05-21):
# This script now routes through the Flower simulation runtime
# (slurm_template_flower.sh → run_one_flower.py). The thesis treats
# Flower as the canonical FL framework; the existing pure-PyTorch
# reference-loop results that were originally produced by this script
# in fl_dermamnist/results/headline/ are preserved unchanged as the
# cross-runtime validation reference (framework=pure-pytorch in their
# back-stamped JSONs).
#
# The canonical PRIMARY headline submitter is now
# submit_flower_C0_baseline.sh, which produces the same FedAvg/FedProx
# pair plus a FedNova arm under one sweep into flower_C0_baseline/.
# This script remains as a 2-algorithm convenience wrapper that
# preserves the historic 20-job shape. Output dir is intentionally kept
# at headline/ per the audit mandate; the per-JSON `framework` field
# distinguishes Flower runs from the existing pure-PyTorch ones in
# place. If you want a clean-slate Flower-only headline directory,
# either run submit_flower_C0_baseline.sh or override the OUT_DIR below.
set -uo pipefail   # NOT -e: a single sbatch failure must not abort the whole sweep

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
OUT_DIR=fl_dermamnist/results/headline
PARTITION=balanced_paired_7_clients     # ← the FedProx-favourable design
MU=0.01                                  # ← from CPU sweep (replace if HPC μ-sweep picks something else)

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

# Pre-flight: refuse to mix runtimes in the same output directory.
# The 20 existing backstamped JSONs in results/headline/ carry
# framework="pure-pytorch"; submitting Flower jobs here would mix
# labels and break downstream analysis filters that compare on the
# framework field (analyse_system_het.py, tables.py).
shopt -s nullglob
existing_jsons=("$REPO_ROOT/$OUT_DIR"/test_at_best_*.json)
shopt -u nullglob
non_flower_found=""
for j in "${existing_jsons[@]}"; do
  if ! grep -q '"framework": "flower-simulation"' "$j" 2>/dev/null; then
    non_flower_found="$j"
    break
  fi
done
if [ -n "$non_flower_found" ]; then
  echo "ERROR: $OUT_DIR already contains a JSON whose framework field is"
  echo "       not 'flower-simulation':"
  echo "         $non_flower_found"
  echo ""
  echo "Submitting this sweep would mix runtimes in the same directory."
  echo "Options:"
  echo "  (a) use submit_flower_C0_baseline.sh (writes to flower_C0_baseline/)"
  echo "  (b) move or delete the existing JSONs before re-running this script"
  echo "  (c) override the output dir explicitly:"
  echo "      OUT_DIR=fl_dermamnist/results/flower_headline_<date> \\"
  echo "        bash fl_dermamnist/scripts/submit_headline.sh"
  exit 3
fi

SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
LOCAL_EPOCHS=20

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_${algo}_mu${mu}_E${LOCAL_EPOCHS}_s${seed}" \
    "$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh" \
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
echo "When done:     bash fl_dermamnist/scripts/check_results.sh"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  echo "Re-run those manually via:"
  echo "  sbatch fl_dermamnist/scripts/slurm_template_flower.sh <algo> <mu> <seed> $LOCAL_EPOCHS $OUT_DIR $PARTITION"
fi
