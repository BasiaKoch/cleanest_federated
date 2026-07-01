#!/bin/bash
# μ-sensitivity sweep for FedProx on the engineered partition (pure-PyTorch).
#
# Defends the thesis's choice of μ=0.01 against the "cherry-picked μ"
# critique. Tests whether the headline pure-PyTorch FedProx advantage
# (Δ=+0.027, p=0.020) is robust across an order of magnitude around μ=0.01.
#
# Why pure-PyTorch (not Flower)
# -----------------------------
# The headline significant result is on pure-PyTorch. The Flower runtime
# attenuates effects ~75% across all conditions, so a Flower μ-sweep would
# just show small flat effects at every μ - not a defense of the headline.
# This sweep adds direct evidence that pure-PyTorch FedProx is not at a
# knife-edge maximum at μ=0.01.
#
# Submitted jobs
# --------------
# 9 jobs: 3 new μ values × 3 seeds.
#   μ values: 0.001, 0.1, 1.0 (Li 2020 swept these + 0.01)
#   seeds:    42, 123, 456  (first 3 of the paired seed protocol)
# μ=0.01 already exists in results/headline/ at all 10 seeds (the headline).
# μ=0.0 (FedAvg) already exists in results/headline/ at all 10 seeds.
#
# Output directory
# ----------------
# fl_dermamnist/results/mu_sweep/
# Filenames follow the standard convention; μ is embedded in the stem
# (e.g., fedprox_mu0.001_E20_s42.json).
#
# Reference: Li et al. 2020, MLSys (FedProx).
#   "We tune μ on a held-out set; best values were in {0.001, 0.01, 0.1, 1.0}."
#
# Usage on HPC login node:
#   cd /home/bk489/federated_clean/cleanest_federated
#   git pull origin main
#   bash infra/slurm/hpc_mu_sweep.sh
#   squeue -u $USER
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TPT="$REPO/infra/slurm/slurm_template_pytorch.sh"
OUT=fl_dermamnist/results/mu_sweep
E=20
PAIRED=balanced_paired_7_clients

# 3 μ values × 3 seeds = 9 jobs
MUS=(0.001 0.1 1.0)
SEEDS=(42 123 456)

cd "$REPO"

# Pure-PyTorch path uses dataloader_generator_seed (from seeding.py) but
# NOT numpy_legacy_seed - server_loop.py does not call np.random.seed per
# (round, cid), so the 32-bit mask is unnecessary on this path.
# We sanity-check by verifying the seeding helper import is in place.
if ! grep -q "from fl_dermamnist.fl.seeding import dataloader_generator_seed" \
        fl_dermamnist/fl/server_loop.py; then
  echo "ERROR: HPC checkout missing seeding.py import in server_loop.py." >&2
  echo "Pull at least commit f6d0fd9 (the seeding refactor)." >&2
  exit 2
fi
if [ ! -f "$TPT" ]; then
  echo "ERROR: pure-PyTorch SLURM template not found at $TPT" >&2
  exit 2
fi

mkdir -p "$OUT" fl_dermamnist/logs

echo "============================================================"
echo " HPC μ-sensitivity sweep (pure-PyTorch FedProx)"
echo "============================================================"
echo " partition: $PAIRED"
echo " μ values:  ${MUS[*]}     (μ=0.01 already in headline/)"
echo " seeds:     ${SEEDS[*]}    (paired with headline seeds 42, 123, 456)"
echo " out_dir:   $OUT"
echo " expected:  9 FedProx jobs"
echo "============================================================"
echo ""

for MU in "${MUS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Submitting FedProx μ=$MU seed=$SEED ..."
        sbatch "$TPT" fedprox "$MU" "$SEED" "$E" "$OUT" "$PAIRED" ""
        sleep 3
    done
done

echo ""
echo "============================================================"
echo " Submitted 9 μ-sweep jobs"
echo " Watch with: squeue -u \$USER"
echo " After completion, analyse the μ-sensitivity with:"
echo "   python fl_dermamnist/analysis/check_mu_sweep.py"
echo "============================================================"
