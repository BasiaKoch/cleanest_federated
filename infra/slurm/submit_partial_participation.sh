#!/bin/bash
# Partial-participation sweep - the second canonical axis of system
# heterogeneity in the FL literature, complementing the local-epoch
# heterogeneity already covered by submit_system_het.sh (C0/C1/C2).
#
# Motivation:
# Every standard FL parameterisation is (K clients, C participation
# fraction, E local epochs, B batch size). Our existing system-het
# sweeps vary E across (C0 uniform, C1 fixed-stragglers, C2
# random-stragglers); the C-axis (variable client participation per
# round) is not yet tested. In real federated deployments, not all
# clients respond every round, so the C=1.0 setting throughout the
# existing sweeps represents the cleanest mathematical baseline but
# the least operationally realistic case.
#
# This sweep tests C=0.5 (half of clients sampled per round, uniformly
# at random) on the engineered partition under uniform local epochs
# E=20 (no local-epoch heterogeneity). The result is interpretable as
# the FedProx-vs-FedAvg comparison under the second axis of system
# heterogeneity, holding all other parameters identical to the C0
# engineered baseline (see submit_flower_C0_baseline.sh).
#
# Mechanism prediction:
# Under partial participation, each round aggregates over a smaller
# random subset (4 of 7 clients per round at C=0.5, computed as
# max(1, round(0.5 * 7)) = 4). Any single client's drift contributes
# a larger fraction of the aggregated update and the aggregation pool
# varies across rounds. The proximal anchor's drift-control mechanism
# is theoretically more useful in this regime (higher per-round
# aggregation variance), but the magnitude of the empirical benefit
# is not strongly predicted by the theory; the experiment is
# descriptive.
#
# Literature anchors:
#   - McMahan et al. (2017) §3.1: original FedAvg paper varies
#     C ∈ {0.1, 0.2, 0.5, 1.0} on MNIST + CIFAR.
#   - Li et al. (2020) §5.3: FedProx tested with partial participation
#     on Synthetic / FEMNIST.
#   - Kairouz et al. (2021) §3.2: lists partial participation as a
#     canonical FL setup parameter.
#
# Hyperparameters: identical to submit_flower_C0_baseline.sh except
# for fraction-fit.
#   • Partition: balanced_paired_7_clients (engineered, as used by
#     the engineered-partition system-het sweeps)
#   • Algorithms: FedAvg (μ=0.0), FedProx (μ=0.01) - FedNova excluded
#     per current thesis scope.
#   • Seeds: 42 123 456 789 999 2024 31337 8675309 161803 271828
#     (same paired-seed protocol as all other sweeps in this thesis).
#   • R = 150 rounds, E = 20 local epochs (uniform), batch = 32,
#     lr = 0.01, momentum = 0.9.
#   • Partial participation: C = 0.5 (4 of 7 clients per round).
#   • --log-update-norms enabled for mechanism evidence.
#
# Compute: 20 jobs (10 seeds × 2 algorithms), ~10 GPU-hours on A100
# at the observed per-job times for the Flower runtime.
#
# Output: fl_dermamnist/results/system_het_partial_C0.5/
#         Files named with the _C0.5 suffix (added automatically by
#         run_one_flower.py when fraction_fit < 1.0).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
LOCAL_EPOCHS=20
SEEDS=(42 123 456 789 999 2024 31337 8675309 161803 271828)
MU=0.01
PARTITION=balanced_paired_7_clients
OUT_DIR=fl_dermamnist/results/system_het_partial_C0.5

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

EXTRA_ARGS="--fraction-fit 0.5 --log-update-norms"

FAILED=()
submit_flower() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_partial_${algo}_mu${mu}_s${seed}" \
    "$REPO_ROOT/infra/slurm/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" "$EXTRA_ARGS"; then
    echo "  FAILED: $algo mu=$mu seed=$seed"
    FAILED+=("$algo $mu $seed")
  fi
  sleep 3
}

for s in "${SEEDS[@]}"; do
  submit_flower fedavg  0.0  "$s"
  submit_flower fedprox $MU  "$s"
done

echo ""
echo "Submitted partial-participation sweep (C=0.5):"
echo "  - 10 seeds × {FedAvg, FedProx} = 20 jobs → $OUT_DIR"
echo "  Total: ~10 GPU-hours on A100."
echo ""
echo "Output files will be named: {algo}_mu{mu}_E20_C0.5_s{seed}.{json,csv,npz}"
echo "Pair this with the existing C=1.0 engineered baseline at"
echo "  fl_dermamnist/results/flower_C0_baseline/"
echo "for the within-runtime C=0.5 vs C=1.0 comparison."

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
