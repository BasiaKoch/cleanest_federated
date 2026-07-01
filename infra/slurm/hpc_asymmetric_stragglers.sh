#!/bin/bash
# Cambridge HPC SLURM submission - Li 2020 §5.2 asymmetric straggler protocol.
#
# WHAT THIS SUBMITS
# -----------------
# 10 jobs: FedAvg WITH --drop-stragglers on the engineered partition
#          under random_stragglers (4 of 7 clients per round at E in {1..19}).
# Outputs to fl_dermamnist/results/system_het_random_asymmetric/.
#
# WHY ONLY FEDAVG (and not FedProx) - RUN-TIME EQUIVALENCE
# ---------------------------------------------------------
# The FedProx arm of the Li 2020 §5.2 comparison is FedProx with
# --drop-stragglers=False on the SAME partition + same straggler schedule.
# Under our deterministic seeding protocol (paired seed -> fixed straggler
# schedule + fixed RNG state for local training), that FedProx run is
# bit-equivalent to our existing FedProx-symmetric-C2 runs already on disk
# at fl_dermamnist/results/system_het_random/. Re-running them would
# burn ~2.5h of HPC compute to reproduce identical numbers. The analysis
# script (check_asymmetric_stragglers.py) reads the FedProx arm from
# system_het_random/ and the FedAvg-drop arm from system_het_random_asymmetric/.
#
# DECOMPOSITION INTERPRETATION
# ----------------------------
# The Li 2020 §5.2 comparison has a known confound: when FedAvg drops
# stragglers but FedProx keeps them, FedProx sees more clients per round.
# Any FedProx win decomposes into:
#   (a) the proximal-anchor mechanism (proximal term stabilising the
#       updates that FedAvg-drop discards), AND
#   (b) FedProx-include's effective sample-size advantage per round.
#
# To distinguish (a) from (b), the analysis script compares THREE arms:
#   Arm 1: FedAvg --drop-stragglers           (new: ~3 clients/round, full work)
#   Arm 2: FedAvg --no-drop = mu=0 include    (existing system_het_random/)
#   Arm 3: FedProx mu=0.01 include            (existing system_het_random/)
# The contrast (Arm 3 - Arm 1) is the literature's headline (+22% claim).
# The contrast (Arm 2 - Arm 1) isolates the partial-work-inclusion effect.
# The contrast (Arm 3 - Arm 2) isolates the proximal-term effect.
#
# WALL-CLOCK CAVEAT
# -----------------
# Flower waits for ALL clients to finish training before invoking
# aggregate_fit; we then DISCARD straggler updates. This faithfully models
# the algorithmic effect on accuracy-per-round but does NOT model
# deadline-bounded wall-clock realism.
#
# USAGE
# -----
# Run AFTER pulling the commit that adds:
#   - fl_dermamnist/fl_flower/strategy_straggler_dropping.py
#   - --drop-stragglers flag in run_one_flower.py
#
#   cd /home/bk489/federated_clean/cleanest_federated
#   git pull origin main
#   bash infra/slurm/hpc_asymmetric_stragglers.sh
#   squeue -u $USER
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TFLW="$REPO/infra/slurm/slurm_template_flower.sh"
OUT=fl_dermamnist/results/system_het_random_asymmetric
E=20
PAIRED=balanced_paired_7_clients

# Paired-seed protocol - same 10 seeds as the headline + system_het sweeps
SEEDS=(42 123 456 789 999 2024 31337 161803 271828 8675309)

cd "$REPO"

# Sanity checks
if ! grep -q "numpy_legacy_seed" fl_dermamnist/fl_flower/client.py; then
  echo "ERROR: HPC checkout missing seed-overflow fix." >&2
  exit 2
fi
if ! grep -q "StragglerDroppingFedAvg" fl_dermamnist/experiments/run_one_flower.py; then
  echo "ERROR: HPC checkout missing --drop-stragglers wiring." >&2
  exit 2
fi
if ! grep -q -- "--drop-stragglers" fl_dermamnist/experiments/run_one_flower.py; then
  echo "ERROR: --drop-stragglers flag not in runner." >&2
  exit 2
fi

mkdir -p "$OUT" fl_dermamnist/logs

echo "============================================================"
echo " HPC asymmetric-straggler protocol (Li 2020 §5.2)"
echo " — FedAvg arm only (10 jobs)"
echo "============================================================"
echo " partition: $PAIRED"
echo " system-het mode: random_stragglers (50% of clients per round)"
echo " seeds: ${SEEDS[*]}"
echo " algo: fedavg WITH --drop-stragglers"
echo " out_dir: $OUT"
echo " (FedProx arm reused from system_het_random/ — bit-equivalent)"
echo "============================================================"
echo ""

RANDOM_EXTRA="--straggler-fraction 0.5 --log-update-norms --system-het-mode random_stragglers --drop-stragglers"

for SEED in "${SEEDS[@]}"; do
    echo "Submitting FedAvg --drop-stragglers seed=$SEED ..."
    sbatch "$TFLW" fedavg 0.0 "$SEED" "$E" "$OUT" "$PAIRED" "$RANDOM_EXTRA"
    sleep 3
done

echo ""
echo "============================================================"
echo " Submitted 10 FedAvg-drop jobs"
echo " Watch with: squeue -u \$USER"
echo " After completion:"
echo "   python fl_dermamnist/analysis/check_asymmetric_stragglers.py"
echo "============================================================"
