#!/bin/bash
# Targeted recovery submission after the Flower client seed-overflow fix.
#
# Run this from the HPC checkout AFTER pulling the commit that contains:
#   - mnist_dermnist/fl/seeding.py
#   - numpy_legacy_seed(...) usage in fl_flower/client.py
#   - numpy_legacy_seed(...) usage in fl_flower/client_fednova.py
#
# This script resubmits only jobs that were missing, provenance-incomplete,
# or known-broken in the synced local audit. It deliberately does NOT rerun
# low-performing non-8675309 FedNova C2 jobs: those may be a real FedNova
# failure mode under extreme random stragglers, not a seed-overflow bug.
set -euo pipefail

REPO=/home/bk489/federated_clean/cleanest_federated
TFLW="$REPO/mnist_dermnist/scripts/slurm_template_flower.sh"
TSYS="$REPO/mnist_dermnist/scripts/slurm_template_system_het.sh"
TFNV="$REPO/mnist_dermnist/scripts/slurm_template_fednova.sh"
E=20
MU=0.01
PAIRED=balanced_paired_7_clients
SPECIALIST=specialist_7_clients
DIRICHLET=dirichlet_alpha01_7_clients
IID=iid_7_clients

cd "$REPO"

if ! grep -q "numpy_legacy_seed" mnist_dermnist/fl_flower/client.py; then
  echo "ERROR: HPC checkout does not contain the seed-overflow fix." >&2
  echo "Pull/sync the latest repo before resubmitting these jobs." >&2
  exit 2
fi

submit_flower() {
  local algo="$1" mu="$2" seed="$3" out="$4" partition="$5" extra="${6:-}"
  sbatch "$TFLW" "$algo" "$mu" "$seed" "$E" "$out" "$partition" "$extra"
  sleep 3
}

submit_sys() {
  local algo="$1" mu="$2" seed="$3" out="$4" mode="$5" extra="$6"
  sbatch "$TSYS" "$algo" "$mu" "$seed" "$E" "$out" "$PAIRED" "$mode" "$extra"
  sleep 3
}

submit_fednova() {
  local seed="$1" out="$2" mode="$3" extra="${4:-}"
  sbatch "$TFNV" "$seed" "$E" "$out" "$PAIRED" "$mode" "$extra"
  sleep 3
}

echo "Submitting required post-seed-fix recovery jobs..."

# 1) Canonical Flower C0 baseline: missing jobs.
submit_flower fedavg  0.0 8675309 mnist_dermnist/results/flower_C0_baseline "$PAIRED" "--log-update-norms"
submit_flower fedprox "$MU" 8675309 mnist_dermnist/results/flower_C0_baseline "$PAIRED" "--log-update-norms"
submit_flower fedprox "$MU" 161803  mnist_dermnist/results/flower_C0_baseline "$PAIRED" "--log-update-norms"
submit_fednova 2024    mnist_dermnist/results/flower_C0_baseline uniform "--log-update-norms"
submit_fednova 8675309 mnist_dermnist/results/flower_C0_baseline uniform "--log-update-norms"
submit_fednova 161803  mnist_dermnist/results/flower_C0_baseline uniform "--log-update-norms"

# 2) IID falsification: seed 8675309 broke under the old NumPy seed path.
submit_flower fedavg  0.0 8675309 mnist_dermnist/results/iid "$IID"
submit_flower fedprox "$MU" 8675309 mnist_dermnist/results/iid "$IID"

# 3) Specialist partition: seed 8675309 broke under the old NumPy seed path.
submit_flower fedavg  0.0 8675309 mnist_dermnist/results/specialist_partition "$SPECIALIST"
submit_flower fedprox "$MU" 8675309 mnist_dermnist/results/specialist_partition "$SPECIALIST"

# 4) Dirichlet alpha=0.1: rerun provenance-missing legacy files plus bad 8675309.
submit_flower fedavg  0.0 123     mnist_dermnist/results/dirichlet_a01 "$DIRICHLET"
submit_flower fedavg  0.0 2024    mnist_dermnist/results/dirichlet_a01 "$DIRICHLET"
submit_flower fedavg  0.0 8675309 mnist_dermnist/results/dirichlet_a01 "$DIRICHLET"
submit_flower fedprox "$MU" 31337   mnist_dermnist/results/dirichlet_a01 "$DIRICHLET"
submit_flower fedprox "$MU" 8675309 mnist_dermnist/results/dirichlet_a01 "$DIRICHLET"

# 5) System heterogeneity C1 fixed stragglers: fill missing jobs and rerun bad 8675309.
FIXED_EXTRA="--straggler-epochs 5 --fixed-straggler-ids 5,6 --log-update-norms"
submit_sys fedavg  0.0 8675309 mnist_dermnist/results/system_het_fixed fixed_stragglers "$FIXED_EXTRA"
submit_sys fedavg  0.0 161803  mnist_dermnist/results/system_het_fixed fixed_stragglers "$FIXED_EXTRA"
submit_sys fedprox "$MU" 42      mnist_dermnist/results/system_het_fixed fixed_stragglers "$FIXED_EXTRA"
submit_sys fedprox "$MU" 999     mnist_dermnist/results/system_het_fixed fixed_stragglers "$FIXED_EXTRA"
submit_sys fedprox "$MU" 2024    mnist_dermnist/results/system_het_fixed fixed_stragglers "$FIXED_EXTRA"
submit_sys fedprox "$MU" 8675309 mnist_dermnist/results/system_het_fixed fixed_stragglers "$FIXED_EXTRA"

# 6) System heterogeneity C2 random stragglers: fill missing jobs and rerun bad 8675309.
RANDOM_EXTRA="--straggler-fraction 0.5 --log-update-norms"
submit_sys fedavg  0.0 123     mnist_dermnist/results/system_het_random random_stragglers "$RANDOM_EXTRA"
submit_sys fedavg  0.0 2024    mnist_dermnist/results/system_het_random random_stragglers "$RANDOM_EXTRA"
submit_sys fedavg  0.0 8675309 mnist_dermnist/results/system_het_random random_stragglers "$RANDOM_EXTRA"
submit_sys fedprox "$MU" 789     mnist_dermnist/results/system_het_random random_stragglers "$RANDOM_EXTRA"
submit_sys fedprox "$MU" 8675309 mnist_dermnist/results/system_het_random random_stragglers "$RANDOM_EXTRA"
submit_sys fedprox "$MU" 161803  mnist_dermnist/results/system_het_random random_stragglers "$RANDOM_EXTRA"

# 7) FedNova C2 random stragglers: fill missing jobs and rerun only seed 8675309.
submit_fednova 123     mnist_dermnist/results/system_het_random_fednova random_stragglers "$RANDOM_EXTRA"
submit_fednova 789     mnist_dermnist/results/system_het_random_fednova random_stragglers "$RANDOM_EXTRA"
submit_fednova 8675309 mnist_dermnist/results/system_het_random_fednova random_stragglers "$RANDOM_EXTRA"

echo ""
echo "Submitted targeted recovery jobs."
echo "After completion, re-run: bash mnist_dermnist/scripts/check_results.sh"
