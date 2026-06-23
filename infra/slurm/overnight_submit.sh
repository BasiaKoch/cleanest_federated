#!/bin/bash
# Overnight unattended submission with SLURM dependency chaining.
#
# Submits all queued experiments with --dependency=afterok:<ray_diag_jid>
# so they ONLY run if the Ray diagnostic succeeds. If Ray is still broken
# (diagnostic returns non-zero), every dependent job is auto-cancelled
# by SLURM and no compute is wasted on guaranteed failures.
#
# Submission order (highest professor-priority first):
#   1. perfect-storm L4         (9 jobs)  — "FedProx clearly wins" experiment
#   2. Li 2020 §5.2 asymmetric  (12 jobs) — asymmetric protocol replication
#   3. node-pinned L4           (6 jobs)  — variance isolation
#   4. extended-rounds L3       (6 jobs)  — convergence fix
#   5. FedNova × equal/unequal-E (12 jobs) — mechanism decomposition
#   6. asymmetric per-client μ  (12 jobs) — Yao 2024 ablation
#   7. μ-sweep ladder           (12 jobs) — μ sensitivity
#
# Total: 69 jobs queued, gated behind one diagnostic.
#
# Pre-requisite: submit ray_diagnostic.sh FIRST and note its job ID.
# This script reads the diagnostic JID from the command line.
#
# Usage
# -----
#   bash infra/slurm/overnight_submit.sh <ray_diag_jobid>
#   DRY_RUN=1 bash infra/slurm/overnight_submit.sh <ray_diag_jobid>
#
# Example:
#   sbatch infra/slurm/ray_diagnostic.sh
#   #   → Submitted batch job 29902842
#   bash infra/slurm/overnight_submit.sh 29902842
set -uo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <ray_diagnostic_jobid>"
    echo ""
    echo "First submit the diagnostic and capture its job ID:"
    echo "  jid=\$(sbatch --parsable infra/slurm/ray_diagnostic.sh)"
    echo "  bash $0 \$jid"
    exit 1
fi

DIAG_JID="$1"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

echo "============================================================"
echo "Overnight submission, gated on Ray diagnostic job $DIAG_JID"
echo "============================================================"
echo "Strategy: every experiment script is submitted with"
echo "  --dependency=afterok:$DIAG_JID"
echo "so jobs auto-cancel if Ray is still broken (no wasted compute)."
echo ""

# All submit scripts are run with the env-var DEP set so each respects
# the dependency. The scripts use sbatch --parsable internally; we pass
# the dependency via SBATCH_DEPENDENCY which sbatch honours globally.
export SBATCH_DEPENDENCY="afterok:$DIAG_JID"
echo "Set SBATCH_DEPENDENCY=$SBATCH_DEPENDENCY"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1: not actually submitting. Would submit:"
fi

SCRIPTS=(
    "submit_fedprox_perfect_storm_L4.sh    # 1. perfect-storm — FedProx clearly wins"
    "submit_li2020_asymmetric_L4.sh        # 2. Li 2020 §5.2 asymmetric protocol"
    "submit_node_pinned_L4.sh              # 3. node-pinned variance isolation"
    "submit_extended_rounds_L3.sh          # 4. extended-rounds L3 convergence fix"
    "submit_fednova_unequal_E.sh           # 5. FedNova × equal/unequal-E"
    "submit_asymmetric_mu_L4.sh            # 6. asymmetric per-client μ (Yao 2024)"
    "submit_mu_sweep_ladder.sh             # 7. μ-sweep ladder"
)

for entry in "${SCRIPTS[@]}"; do
    # entry = "filename.sh   # comment"
    script=$(echo "$entry" | awk '{print $1}')
    comment=$(echo "$entry" | sed 's/^[^#]*//')
    full_path="$REPO_ROOT/infra/slurm/$script"
    if [ ! -x "$full_path" ]; then
        echo "  SKIP (not executable or not found): $script"
        continue
    fi
    echo ""
    echo "======================================================"
    echo "Submitting:  $script"
    echo "$comment"
    echo "======================================================"
    if [ "$DRY_RUN" = "1" ]; then
        DRY_RUN=1 bash "$full_path"
    else
        bash "$full_path"
    fi
    sleep 2
done

unset SBATCH_DEPENDENCY

echo ""
echo "============================================================"
echo "Overnight submission complete."
echo "============================================================"
echo ""
echo "Status check commands:"
echo "  squeue -u \$USER --format='%.10i %.45j %.8T %.10M %.20R' | head -30"
echo "  sacct -u \$USER --starttime now --format=JobID,JobName%35,State,ExitCode,Elapsed | head -40"
echo ""
echo "After Ray diagnostic finishes (~5 min):"
echo "  cat fl_dermamnist/logs/ray_diag_${DIAG_JID}.out"
echo ""
echo "If ray_diag shows ✅, all dependent jobs proceed. If ❌, they auto-cancel."
echo "Either way, you can go to bed; SLURM handles the rest."
