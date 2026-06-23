#!/bin/bash
# Mechanism probe: τ-clipping under random-straggler FedNova on the
# balanced-paired-7-clients partition. Tests the hypothesis that the
# 1/τ amplification term is the root cause of the 3-of-10-seeds
# collapse observed in submit_fednova_system_het.sh.
#
# Protocol matches submit_fednova_system_het.sh EXACTLY except for the
# added `--tau-clip-min 320` flag. `tau_i` in FedNova is the number of
# local SGD STEPS (= local_epochs × batches_per_epoch), not epochs;
# for balanced_paired_7_clients (~1001 samples/client) with batch=32,
# batches_per_epoch ≈ 32, so 320 ≈ 10 epochs of work — i.e. E_max // 2
# expressed in the correct step units. The clamp replaces tau_i with
# 320 in the FedNova normaliser denominator when tau_i < 320; partial
# parameter deltas still enter aggregation unchanged.
#
# Seeds tested: the Stage-2 pilot set (FEDNOVA_RANDOM_TAU_THESIS_PLAN.md §6):
# 2 born-collapse seeds (31337, 271828), 2 transient-peak-then-collapse seeds
# (42, 123), and the best-baseline seed (8675309). NB under FedNova-random ALL
# baseline seeds are < 0.40 (best 0.364); the success criterion is lifting the
# set toward the FedAvg band (~0.49) with no trivial-predictor endpoints.
#
# Output dir: fl_dermamnist/results/system_het_random_fednova_tauclip/
# Compute: ~5 GPU-h on A100 (5 seeds × ~1 GPU-h each).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
# tau_i is the local SGD step count, not the epoch count. With
# balanced_paired_7_clients (~1001 samples/client) and batch=32,
# batches_per_epoch ≈ 32, so a 10-epoch clamp ≈ 320 steps.
TAU_CLIP_MIN=320            # ≈ E_max/2 epochs in step units (10 × 32 batches)
SEEDS=(42 123 8675309 31337 271828)   # Stage-2 pilot set (plan §6)
PARTITION=balanced_paired_7_clients

FAILED=()
submit() {
  local seed="$1" out="$2" sh_mode="$3" extra="$4"
  if ! sbatch \
    --job-name="mn_fn_tauclip_${sh_mode}_s${seed}" \
    "$REPO_ROOT/infra/slurm/slurm_template_fednova.sh" \
    "$seed" "$LOCAL_EPOCHS" "$out" "$PARTITION" "$sh_mode" "$extra"; then
    echo "  FAILED to submit: seed=$seed sh=$sh_mode"
    FAILED+=("$seed $sh_mode")
  fi
  sleep 3
}

OUT=fl_dermamnist/results/system_het_random_fednova_tauclip
mkdir -p "$REPO_ROOT/$OUT"
for s in "${SEEDS[@]}"; do
  # NOTE: slurm_template_fednova.sh hardcodes --batch-size 10 --momentum 0.0,
  # but the collapsed-seed baseline runs used batch=32, momentum=0.9 (see
  # fl_dermamnist/results/system_het_random_fednova/test_at_best_*.json).
  # Argparse uses the LAST occurrence, so the extras below override the
  # template defaults and match the baseline protocol exactly.
  submit "$s" "$OUT" random_stragglers \
    "--batch-size 32 --momentum 0.9 --straggler-fraction 0.5 --log-update-norms --tau-clip-min ${TAU_CLIP_MIN}"
done

echo ""
echo "Submitted FedNova τ-clip probe sweep:"
echo "  - random_stragglers × ${#SEEDS[@]} seeds (pilot: 2 born-collapse, 2 transient, 1 best)"
echo "  - τ_clip_min = ${TAU_CLIP_MIN} steps  (= 10 epochs × 32 batches; ≈ E_max/2 in step units)"
echo "  - output: $OUT/"
echo ""
echo "Success criterion: τ-clip lifts the pilot seeds out of majority-class"
echo "collapse (final-round macro-F1 toward the FedAvg band ~0.49, 0 trivial"
echo "endpoints), implicating the 1/τ-amplification (direction) mechanism."
echo ""
echo "When complete:"
echo "  PYTHONPATH=. python fl_dermamnist/analysis/peek_system_het_progress.py \\"
echo "      --dir fl_dermamnist/results/system_het_random_fednova_tauclip"
echo "Monitor with:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
