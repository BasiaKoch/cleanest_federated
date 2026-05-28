#!/bin/bash
# Status of all thesis experiment directories.
#
# Counts per-sweep completion of test_at_best_*.json against the documented
# 10-paired-seed standard. Optional: SLURM queue + recent failures.
set -euo pipefail

# Resolve the repo root relative to this script's location.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

count_complete() {
  local dir="$1"
  if [ -d "$dir" ]; then
    ls "$dir"/test_at_best_*.json 2>/dev/null | wc -l
  else
    echo 0
  fi
}

count_history() {
  local dir="$1"
  if [ -d "$dir" ]; then
    ls "$dir"/history_*.csv 2>/dev/null | wc -l
  else
    echo 0
  fi
}

count_centralised() {
  local dir="$1"
  if [ -d "$dir" ]; then
    ls "$dir"/centralised_seed*.json 2>/dev/null | wc -l
  else
    echo 0
  fi
}

echo "=== mnist_dermnist experiment status ==="
echo ""
# Per-sweep tuples: <results-subdir>  <expected job count>  <runtime tag>
# Expected counts assume the 10-paired-seed standard (10×2 = 20) unless noted.
SWEEPS=(
  "headline                       20  pure-PyTorch  engineered, primary headline"
  "iid                            20  pure-PyTorch  IID mechanism-null"
  "dirichlet_a01                  20  pure-PyTorch  Dirichlet(α=0.1)"
  "flower_C0_baseline             30  Flower        engineered, runtime replication (+FedNova=10)"
  "flower_C0_iid_baseline         30  Flower        IID (+FedNova=10)"
  "specialist_partition           20  Flower        specialist falsification probe"
  "system_het_fixed               20  Flower        S1 fixed-straggler"
  "system_het_random              20  Flower        S2 random-straggler"
  "system_het_iid_fixed           20  Flower        IID + fixed-straggler control"
  "system_het_iid_random          20  Flower        IID + random-straggler control"
  "system_het_random_asymmetric   20  Flower        asymmetric-straggler protocol"
  "system_het_random_fednova      10  Flower        FedNova comparator"
  "system_het_partial_C0.5        20  Flower        partial participation C=0.5"
  "mu_sensitivity_flower          50  Flower        μ ∈ {0.001,0.01,0.1,1.0} × 10 + 10 FedAvg"
  "mu_sweep                        6  pure-PyTorch  legacy μ sweep, n=3 seeds (superseded)"
  "arch_ablation_bn                6  Flower        ARCHIVED — BN ablation, not in thesis"
)

total_done=0
total_expected=0
for entry in "${SWEEPS[@]}"; do
  # Split into name / expected / runtime / description
  name=$(echo "$entry"      | awk '{print $1}')
  exp=$(echo "$entry"       | awk '{print $2}')
  runtime=$(echo "$entry"   | awk '{print $3}')
  desc=$(echo "$entry"      | cut -d' ' -f4-)
  done=$(( $(count_complete "mnist_dermnist/results/$name") ))
  hist=$(( $(count_history  "mnist_dermnist/results/$name") ))
  status=" "
  if   [ "$done" -ge "$exp" ]; then status="✓"
  elif [ "$done" -eq 0       ]; then status=" "
  else                              status="…"
  fi
  printf "  %s  %-32s : %3d/%-3d  (hist=%3d)  [%-12s] %s\n" \
    "$status" "$name" "$done" "$exp" "$hist" "$runtime" "$desc"
  total_done=$((total_done + done))
  total_expected=$((total_expected + exp))
done

# Centralised reference is structured differently (no test_at_best_*.json).
cent_done=$(( $(count_centralised "mnist_dermnist/results/centralised") ))
printf "  %s  %-32s : %3d/10   (—)        [%-12s] %s\n" \
  "$( [ "$cent_done" -ge 10 ] && echo ✓ || echo " " )" \
  "centralised" "$cent_done" "centralised" \
  "reference performance ceiling (centralised_seed*.json)"

echo ""
printf "  ── total: %d / %d test_at_best JSONs across all sweeps ──\n" \
  "$total_done" "$total_expected"

# SLURM-side status (no-op when not on HPC)
if command -v squeue >/dev/null 2>&1; then
  echo ""
  echo "=== Queue ==="
  squeue -u "$USER" --format="%.12i %.40j %.2t %.10M" | head -25
  echo ""
  echo "=== Recent failures (today) ==="
  sacct -u "$USER" --starttime="$(date -d 'today' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)" \
        --format=JobID,JobName%40,State,Elapsed,ExitCode 2>/dev/null \
    | grep -E "FAILED|TIMEOUT" | head -10 || echo "  none"
fi

echo ""
echo "To run thesis-ready analysis after sweeps complete:"
echo "  bash mnist_dermnist/scripts/analyse_all.sh"
echo ""
echo "Per-directory provenance, partitions, and git-commit fields:"
echo "  mnist_dermnist/results/PROVENANCE_AUDIT.md"
