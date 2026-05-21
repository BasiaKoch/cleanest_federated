#!/bin/bash
# Status of headline + μ sweep + E sweep runs.
set -euo pipefail

# Resolve the repo root relative to this script's location so the script
# works from any checkout (CSD3, laptop, CI). check_results.sh lives at
# <REPO_ROOT>/mnist_dermnist/scripts/, so go two levels up.
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

echo "=== mnist_dermnist run status ==="
echo ""
# Track per-sweep progress. Tuples: <results-subdir> <expected job count>
SWEEPS=(
  "headline                       20"   # 10 seeds × {FedAvg, FedProx}
  "mu_sweep                       18"   # 3 FedAvg + 5 μ × 3 seeds (Li 2020 grid)
  "e_sweep                        30"   # 5 E × 2 algos × 3 seeds
  "headline_flower_verify          4"   # 2 seeds × {FedAvg, FedProx} via Flower
  "flower_C0_baseline             30"   # 10 seeds × {FedAvg, FedProx, FedNova}
  "system_het_fixed               20"   # C1: 10 seeds × {FedAvg, FedProx}
  "system_het_random              20"   # C2: 10 seeds × {FedAvg, FedProx}
  "system_het_random_fednova      10"   # C2: 10 seeds × FedNova
  "iid                            20"   # IID falsification
  "dirichlet_a01                  20"   # Dirichlet-α=0.1 robustness
  "class_weighted_baseline        10"   # FedAvg + CW-CE × 10 seeds
)

total_done=0
total_expected=0
for entry in "${SWEEPS[@]}"; do
  # Split on whitespace
  name=$(echo "$entry" | awk '{print $1}')
  exp=$(echo "$entry" | awk '{print $2}')
  # `wc -l` on macOS prints whitespace-padded counts; coerce to int via $(())
  done=$(( $(count_complete "mnist_dermnist/results/$name") ))
  hist=$(( $(count_history  "mnist_dermnist/results/$name") ))
  status=" "
  if   [ "$done" -ge "$exp" ]; then status="✓"
  elif [ "$done" -eq 0       ]; then status=" "
  else                              status="…"
  fi
  # Pad name to 30 chars for column alignment
  printf "  %s  %-30s : %3d/%-3d complete   (%3d history CSVs)\n" \
    "$status" "$name" "$done" "$exp" "$hist"
  total_done=$((total_done + done))
  total_expected=$((total_expected + exp))
done

echo ""
printf "  ── total: %d / %d test_at_best JSONs across all sweeps ──\n" \
  "$total_done" "$total_expected"

echo ""
echo "=== Queue ==="
squeue -u "$USER" --format="%.12i %.40j %.2t %.10M" | head -25
echo ""
echo "=== Recent failures ==="
sacct -u "$USER" --starttime="$(date -d 'today' +%Y-%m-%d)" \
      --format=JobID,JobName%40,State,Elapsed,ExitCode 2>/dev/null \
  | grep -E "FAILED|TIMEOUT" | head -10 || echo "  none"

echo ""
echo "To analyze (after all 20 headline complete):"
echo "  PYTHONPATH=. python -m mnist_dermnist.analysis.tables --results-dir mnist_dermnist/results/headline --E 20"
echo "  PYTHONPATH=. python -m mnist_dermnist.analysis.plots  --results-dir mnist_dermnist/results/headline --E 20"
