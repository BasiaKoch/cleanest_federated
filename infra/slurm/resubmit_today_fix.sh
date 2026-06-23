#!/bin/bash
# Recovery script for 9 jobs that hit CUDA-busy crashes during today's
# resubmission batch (2026-05-22). Each crash exited code 1:0 in 18-36
# seconds during the runner's first .to(device) call — transient HPC GPU
# contention, not a code bug.
#
# Missing jobs grouped by sweep:
#
#   flower_C0_baseline (6 jobs):
#     - fedavg  s8675309
#     - fedprox s8675309, s161803
#     - fednova s2024, s8675309, s161803
#
#   mu_sweep (3 jobs, FedAvg sanity baselines):
#     - fedavg μ=0.0 s42, s123, s456
#
# Each retry has an independent chance of hitting the same CUDA-busy
# pattern; if any of these fail again, re-run this script. Sleep between
# sbatches is intentional (3s) — gives the scheduler time to spread
# allocations across nodes instead of bursting onto one node.
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TFLW="$REPO/infra/slurm/slurm_template_flower.sh"
TFNV="$REPO/infra/slurm/slurm_template_fednova.sh"

# --- flower_C0_baseline FedAvg + FedProx (3 jobs) ---
sbatch "$TFLW" fedavg  0.0  8675309 20 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients "--log-update-norms" ; sleep 3
sbatch "$TFLW" fedprox 0.01 8675309 20 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients "--log-update-norms" ; sleep 3
sbatch "$TFLW" fedprox 0.01 161803  20 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients "--log-update-norms" ; sleep 3

# --- flower_C0_baseline FedNova (3 jobs) ---
sbatch "$TFNV" 2024    20 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform "--log-update-norms" ; sleep 3
sbatch "$TFNV" 8675309 20 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform "--log-update-norms" ; sleep 3
sbatch "$TFNV" 161803  20 fl_dermamnist/results/flower_C0_baseline balanced_paired_7_clients uniform "--log-update-norms" ; sleep 3

# --- mu_sweep FedAvg sanity baselines (3 jobs) ---
sbatch "$TFLW" fedavg 0.0 42  20 fl_dermamnist/results/mu_sweep balanced_paired_7_clients ; sleep 3
sbatch "$TFLW" fedavg 0.0 123 20 fl_dermamnist/results/mu_sweep balanced_paired_7_clients ; sleep 3
sbatch "$TFLW" fedavg 0.0 456 20 fl_dermamnist/results/mu_sweep balanced_paired_7_clients ; sleep 3

echo ""
echo "Dispatched 9 resubmissions:"
echo "  - 6 to flower_C0_baseline (FedAvg s8675309, FedProx s8675309/s161803,"
echo "                             FedNova s2024/s8675309/s161803)"
echo "  - 3 to mu_sweep (FedAvg μ=0.0 baselines s42/s123/s456)"
