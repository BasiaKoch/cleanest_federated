#!/bin/bash
# Rescue script: resubmit μ-sweep jobs that hit CUDA-busy failures.
#
# Background: the original hpc_mu_sweep.sh submission landed 3 of 9 jobs
# successfully and 6 hit a transient CUDA-busy error at
# `model_builder().to(device)` (the first GPU access). The SLURM template's
# 3-attempt retry loop with 90s sleep was not enough to recover — those
# jobs landed on a heavily contended Ampere node.
#
# This script is idempotent: it checks the disk and resubmits only the
# (μ, seed) combinations whose test_at_best_*.json doesn't exist yet.
#
# Usage on HPC login node:
#   cd /home/bk489/federated_clean/cleanest_federated
#   git pull origin main
#   bash infra/slurm/hpc_mu_sweep_rescue.sh
#   squeue -u $USER
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TPT="$REPO/infra/slurm/slurm_template_pytorch.sh"
OUT=fl_dermamnist/results/mu_sweep
E=20
PAIRED=balanced_paired_7_clients

MUS=(0.001 0.1 1.0)
SEEDS=(42 123 456)

cd "$REPO"

if [ ! -f "$TPT" ]; then
  echo "ERROR: pure-PyTorch SLURM template not found at $TPT" >&2
  exit 2
fi

mkdir -p "$OUT" fl_dermamnist/logs

echo "============================================================"
echo " μ-sweep RESCUE — resubmit only failed/missing combinations"
echo "============================================================"
echo ""

# Check what's already on disk and queue what isn't
SUBMITTED=0
SKIPPED=0
for MU in "${MUS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Filename convention from run_one.py: fedprox_mu{MU}_E20_s{SEED}.json
        OUT_JSON="$OUT/test_at_best_fedprox_mu${MU}_E20_s${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo "  SKIP    μ=$MU seed=$SEED  (already on disk: $OUT_JSON)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "  RESUBMIT μ=$MU seed=$SEED  (no file at $OUT_JSON)"
            sbatch "$TPT" fedprox "$MU" "$SEED" "$E" "$OUT" "$PAIRED" ""
            sleep 3
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

echo ""
echo "============================================================"
echo " Done: submitted=$SUBMITTED  skipped=$SKIPPED"
echo " Watch with: squeue -u \$USER"
echo " After completion:"
echo "   python fl_dermamnist/analysis/check_mu_sweep.py"
echo "============================================================"
