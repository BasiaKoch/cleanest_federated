#!/bin/bash
# Status of headline + μ sweep + E sweep runs.
set -euo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
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
echo "headline   : $(count_complete mnist_dermnist/results/headline)/20 complete  ($(count_history mnist_dermnist/results/headline) history CSVs)"
echo "mu_sweep   : $(count_complete mnist_dermnist/results/mu_sweep)/15 complete"
echo "e_sweep    : $(count_complete mnist_dermnist/results/e_sweep)/30 complete"
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
