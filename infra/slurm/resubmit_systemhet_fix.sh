#!/bin/bash
# One-shot recovery: resubmit the 12 system-het jobs that were broken by a
# terminal line-split during yesterday's interactive resubmission attempt.
#
# Background: a copy-paste of multi-line sbatch commands into the user's
# HPC shell was line-wrapped by the terminal at column ~80, splitting each
# sbatch invocation into two physical commands. The first half submitted
# successfully but WITHOUT the system-het-mode arg and the straggler
# config; the second half ("fixed_stragglers"/"random_stragglers" with the
# extras) was parsed as a missing command. The 12 affected JobIDs were:
#
#   29504637–29504641  system_het_fixed (C1)  — 5 jobs
#   29504643–29504647  system_het_random (C2) — 5 jobs
#   29504648, 29504649 system_het_random_fednova — 2 jobs
#
# All 12 were scancel'd before this script ran. This file re-submits them
# correctly with proper system-het arguments.
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TSH="$REPO/infra/slurm/slurm_template_system_het.sh"
TFNV="$REPO/infra/slurm/slurm_template_fednova.sh"

EXTRA_C1="--straggler-epochs 5 --fixed-straggler-ids 5,6 --log-update-norms"
EXTRA_C2="--straggler-fraction 0.5 --log-update-norms"

# --- system_het_fixed (C1, 5 jobs) ---
sbatch "$TSH" fedprox 0.01 42      20 fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
sbatch "$TSH" fedprox 0.01 999     20 fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
sbatch "$TSH" fedprox 0.01 2024    20 fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
sbatch "$TSH" fedavg  0.0  8675309 20 fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"
sbatch "$TSH" fedavg  0.0  161803  20 fl_dermamnist/results/system_het_fixed balanced_paired_7_clients fixed_stragglers "$EXTRA_C1"

# --- system_het_random (C2 FedAvg+FedProx, 5 jobs) ---
sbatch "$TSH" fedavg  0.0  123     20 fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
sbatch "$TSH" fedprox 0.01 789     20 fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
sbatch "$TSH" fedavg  0.0  2024    20 fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
sbatch "$TSH" fedprox 0.01 8675309 20 fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"
sbatch "$TSH" fedprox 0.01 161803  20 fl_dermamnist/results/system_het_random balanced_paired_7_clients random_stragglers "$EXTRA_C2"

# --- system_het_random_fednova (FedNova arm of C2, 2 jobs) ---
sbatch "$TFNV" 123 20 fl_dermamnist/results/system_het_random_fednova balanced_paired_7_clients random_stragglers "$EXTRA_C2"
sbatch "$TFNV" 789 20 fl_dermamnist/results/system_het_random_fednova balanced_paired_7_clients random_stragglers "$EXTRA_C2"

echo ""
echo "Dispatched 12 system-het resubmissions:"
echo "  - 5 to system_het_fixed (C1, fixed stragglers C5/C6 at E=5)"
echo "  - 5 to system_het_random (C2, random stragglers fraction=0.5)"
echo "  - 2 to system_het_random_fednova (FedNova C2)"
echo ""
echo "Verify with:  squeue -u \$USER --format='%j' | grep -E 'fixed|random' | sort -u"
