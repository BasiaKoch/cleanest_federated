#!/bin/bash
# Remove any result directory that is incomplete (no global_test_metrics.json).
# Safe: only deletes directories that look like aborted FL runs.
set -euo pipefail

RESULTS_DIR="${1:-results}"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "Results directory '$RESULTS_DIR' does not exist."
  exit 1
fi

echo "Scanning '$RESULTS_DIR' for incomplete experiment directories..."

removed=0
kept=0
for d in "$RESULTS_DIR"/*_E*_s*; do
  [ -d "$d" ] || continue
  if [ -f "$d/global_test_metrics.json" ]; then
    kept=$((kept + 1))
  else
    echo "  REMOVING: $d  (no global_test_metrics.json)"
    rm -rf "$d"
    removed=$((removed + 1))
  fi
done

# Also cleanup empty ablations subdirs
for d in "$RESULTS_DIR/ablations"/*/abl_*_s*; do
  [ -d "$d" ] || continue
  if [ ! -f "$d/global_test_metrics.json" ]; then
    echo "  REMOVING: $d  (incomplete ablation)"
    rm -rf "$d"
    removed=$((removed + 1))
  fi
done

# Cleanup stray .err/.out logs older than 7 days
if [ -d "logs" ]; then
  find logs -maxdepth 1 -type f -name "*.err" -size 0 -delete 2>/dev/null || true
  find logs -maxdepth 1 -type f -name "*.out" -size 0 -delete 2>/dev/null || true
fi

echo ""
echo "Cleanup complete: removed $removed incomplete directories, kept $kept successful runs."
