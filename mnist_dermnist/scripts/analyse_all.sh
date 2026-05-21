#!/bin/bash
# analyse_all.sh — dispatch analysis for every populated results directory.
#
# Workflow:
#   1. Scan mnist_dermnist/results/<sweep>/ for every directory that contains
#      at least one test_at_best_*.json.
#   2. For each populated FL result directory, run analysis.tables.
#   3. If flower_C0_baseline/ AND system_het_random/ are populated, run the
#      system-heterogeneity analyser (which also reads system_het_fixed and
#      system_het_random_fednova if they exist).
#   4. If headline_flower_verify/ is populated, run compare_equivalence_full_scale.
#   5. Print a final summary listing which sweeps were analysed and which
#      are still missing.
#
# The script is idempotent and safe to re-run as HPC jobs trickle in.
#
# Exit code is 0 even if some sweeps are missing — the summary makes that
# explicit, but a missing sweep is not an error.
set -uo pipefail

# Resolve repo root relative to this script (script lives in mnist_dermnist/scripts/).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
RESULTS="mnist_dermnist/results"

# Every directory we know how to analyse, with its dedicated analyser column:
#   "<dir-under-results>  <analyser-tag>"
# analyser tags:
#   tables      → analysis.tables
#   syshet      → analyse_system_het.py (handled separately)
#   equiv       → compare_equivalence_full_scale.py (handled separately)
SWEEPS=(
  "headline                  tables"
  "mu_sweep                  tables"
  "e_sweep                   tables"
  "iid                       tables"
  "dirichlet_a01             tables"
  "class_weighted_baseline   tables"
  "flower_C0_baseline        tables"
  "system_het_fixed          tables"
  "system_het_random         tables"
  "system_het_random_fednova tables"
  "headline_flower_verify    tables"
)

is_populated() {
  local dir="$1"
  [ -d "$dir" ] || return 1
  ls "$dir"/test_at_best_*.json >/dev/null 2>&1
}

analysed=()
skipped=()
failed=()

echo "=== Per-sweep analysis (analysis.tables) ==="
echo ""
for entry in "${SWEEPS[@]}"; do
  name=$(echo "$entry" | awk '{print $1}')
  dir="$RESULTS/$name"
  if is_populated "$dir"; then
    n_files=$(ls "$dir"/test_at_best_*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "→ Analysing $name ($n_files JSON files)..."
    if PYTHONPATH=. python -m mnist_dermnist.analysis.tables \
         --results-dir "$dir" --E 20 \
         >"$dir/analysis_log.txt" 2>&1; then
      analysed+=("$name")
      echo "  ✓ wrote $dir/analysis/paired_stats.json + final_test_table.csv"
    else
      failed+=("$name")
      echo "  ✗ FAILED (see $dir/analysis_log.txt)"
    fi
  else
    skipped+=("$name")
  fi
done

echo ""
echo "=== System-heterogeneity analysis (cross-condition H2) ==="
echo ""
if is_populated "$RESULTS/flower_C0_baseline" \
   && is_populated "$RESULTS/system_het_random"; then
  echo "→ Running analyse_system_het.py (C0 + C1/C2 + FedNova arms)..."
  if PYTHONPATH=. python mnist_dermnist/results/thesis_ready_system_het/scripts/analyse_system_het.py \
       >"$RESULTS/thesis_ready_system_het_log.txt" 2>&1; then
    echo "  ✓ wrote $RESULTS/thesis_ready_system_het/data/summary_statistics.json"
    analysed+=("system_het_pipeline")
  else
    echo "  ✗ FAILED (see $RESULTS/thesis_ready_system_het_log.txt)"
    failed+=("system_het_pipeline")
  fi
else
  echo "  – skipped: requires flower_C0_baseline/ AND system_het_random/ to be populated"
  skipped+=("system_het_pipeline")
fi

echo ""
echo "=== Cross-runtime equivalence check ==="
echo ""
if is_populated "$RESULTS/headline_flower_verify"; then
  echo "→ Running compare_equivalence_full_scale..."
  if PYTHONPATH=. python -m mnist_dermnist.experiments.compare_equivalence_full_scale \
       >"$RESULTS/equivalence_full_scale_log.txt" 2>&1; then
    echo "  ✓ wrote $RESULTS/thesis_ready/data/equivalence_full_scale.json"
    analysed+=("equivalence_full_scale")
  else
    echo "  ✗ FAILED (see $RESULTS/equivalence_full_scale_log.txt)"
    failed+=("equivalence_full_scale")
  fi
else
  echo "  – skipped: headline_flower_verify/ is not populated"
  skipped+=("equivalence_full_scale")
fi

echo ""
echo "=== Headline extras (sign / Hodges-Lehmann / LOSO / Holm / per-class) ==="
echo ""
if is_populated "$RESULTS/headline"; then
  echo "→ Running analyse_extra_statistics on headline..."
  PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/analyse_extra_statistics.py \
    --results-dir "$RESULTS/headline" \
    >"$RESULTS/headline_extra_stats_log.txt" 2>&1 \
    && echo "  ✓ wrote thesis_ready/data/extra_statistics.json" \
    || echo "  ✗ FAILED (see $RESULTS/headline_extra_stats_log.txt)"
  echo "→ Running plot_per_class_delta on headline..."
  PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/plot_per_class_delta.py \
    --results-dir "$RESULTS/headline" \
    >"$RESULTS/headline_per_class_log.txt" 2>&1 \
    && echo "  ✓ wrote thesis_ready/figures/per_class_delta.{png,pdf}" \
    || echo "  ✗ FAILED (see $RESULTS/headline_per_class_log.txt)"
  echo "→ Running analyse_communication_metrics on headline..."
  PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/analyse_communication_metrics.py \
    --results-dir "$RESULTS/headline" \
    >"$RESULTS/headline_comm_log.txt" 2>&1 \
    && echo "  ✓ wrote thesis_ready/data/communication_metrics.json" \
    || echo "  ✗ FAILED (see $RESULTS/headline_comm_log.txt)"
fi

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""
if [ ${#analysed[@]} -gt 0 ]; then
  echo "Analysed (${#analysed[@]}):"
  for x in "${analysed[@]}"; do echo "  ✓ $x"; done
fi
if [ ${#skipped[@]} -gt 0 ]; then
  echo ""
  echo "Skipped — not yet populated (${#skipped[@]}):"
  for x in "${skipped[@]}"; do echo "  – $x"; done
fi
if [ ${#failed[@]} -gt 0 ]; then
  echo ""
  echo "FAILED (${#failed[@]}):"
  for x in "${failed[@]}"; do echo "  ✗ $x"; done
fi

echo ""
echo "To re-run after more HPC jobs land:"
echo "  bash mnist_dermnist/scripts/analyse_all.sh"
echo ""
echo "To inspect overall HPC progress:"
echo "  bash mnist_dermnist/scripts/check_results.sh"
