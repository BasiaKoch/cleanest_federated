#!/bin/bash
# Cambridge HPC SLURM equivalent of runpod_arch_ablation_bn.sh.
#
# Submits 6 jobs (3 seeds * 2 algos) for the BatchNorm architecture ablation
# on the engineered balanced_paired_7_clients partition. All runs use
# --model-variant bn so the runner instantiates DermMNISTCNN_BN.
#
# Prerequisite: the HPC checkout must contain commit 44a068d or later
# (adds DermMNISTCNN_BN + --model-variant flag). This script aborts if
# the flag is absent from the runner's CLI.
#
# Outputs go to fl_dermamnist/results/arch_ablation_bn/ - a separate
# directory from the GroupNorm headline so the two variants cannot be
# mixed. Filenames also carry an '_arch-bn' tag.
#
# Usage on HPC login node:
#   cd /home/bk489/federated_clean/cleanest_federated
#   git pull origin main
#   bash infra/slurm/hpc_arch_ablation_bn.sh
#   squeue -u $USER   # to watch
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TFLW="$REPO/infra/slurm/slurm_template_flower.sh"
OUT=fl_dermamnist/results/arch_ablation_bn
E=20
MU=0.01
PAIRED=balanced_paired_7_clients
EXTRA_ARGS="--model-variant bn --log-update-norms"

cd "$REPO"

# Sanity check 1: seeding fix must be present (required for s8675309)
if ! grep -q "numpy_legacy_seed" fl_dermamnist/runtimes/flower/client.py; then
  echo "ERROR: HPC checkout missing the seed-overflow fix." >&2
  echo "Pull latest main before submitting these jobs." >&2
  exit 2
fi

# Sanity check 2: --model-variant flag must be present
if ! grep -q -- "--model-variant" fl_dermamnist/experiments/run_one_flower.py; then
  echo "ERROR: HPC checkout missing --model-variant flag." >&2
  echo "Pull at least commit 44a068d before submitting these jobs." >&2
  exit 2
fi

mkdir -p "$OUT" fl_dermamnist/logs

echo "============================================================"
echo " HPC BN architecture ablation (Proposal A)"
echo "============================================================"
echo " variant:   DermMNISTCNN_BN (BatchNorm2d, no GroupNorm)"
echo " partition: $PAIRED"
echo " seeds:     42, 123, 456"
echo " algos:     fedavg, fedprox"
echo " extra:     $EXTRA_ARGS"
echo " out_dir:   $OUT"
echo "============================================================"
echo ""

# 3 seeds * 2 algos = 6 jobs
for SEED in 42 123 456; do
    echo "Submitting BN ablation: algo=fedavg seed=$SEED ..."
    sbatch "$TFLW" fedavg  0.0   "$SEED" "$E" "$OUT" "$PAIRED" "$EXTRA_ARGS"
    sleep 3

    echo "Submitting BN ablation: algo=fedprox seed=$SEED ..."
    sbatch "$TFLW" fedprox "$MU" "$SEED" "$E" "$OUT" "$PAIRED" "$EXTRA_ARGS"
    sleep 3
done

echo ""
echo "=== Submitted 6 BN ablation jobs ==="
echo "Watch with: squeue -u \$USER"
echo "After completion, compare to GN headline with:"
echo "  python fl_dermamnist/analysis/check_arch_ablation.py"
