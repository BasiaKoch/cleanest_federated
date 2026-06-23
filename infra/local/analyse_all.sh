#!/bin/bash
# analyse_all.sh — dispatch analysis for every populated thesis result
# directory in the current experiment matrix.
#
# Workflow:
#   1. For every populated FL result directory (test_at_best_*.json present),
#      run `analysis.tables` to produce per-seed summaries and paired stats.
#   2. Run the thesis-ready statistical-heterogeneity analyser if the
#      engineered + system-het inputs are populated.
#   3. Run the cross-runtime equivalence audit if pure-PyTorch headline and
#      Flower baseline are both populated.
#   4. Run the thesis-ready extras (per-class delta, communication metrics,
#      extra statistics, P/R/F1 decomposition) on the headline if populated.
#   5. Print a final summary listing which sweeps were analysed and which
#      are still missing.
#
# The script is idempotent and safe to re-run as HPC jobs trickle in.
# Exit code is 0 even if some sweeps are missing.
set -uo pipefail

# Resolve repo root relative to this script's location.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
RESULTS="fl_dermamnist/results"

# Every directory we know how to analyse with the legacy `analysis.tables`
# entry point. Per-directory thesis-ready scripts live under
# fl_dermamnist/analysis/ and are dispatched separately below.
SWEEPS=(
  "headline"
  "iid"
  "dirichlet_a01"
  "flower_C0_baseline"
  "flower_C0_iid_baseline"
  "specialist_partition"
  "system_het_fixed"
  "system_het_random"
  "system_het_iid_fixed"
  "system_het_iid_random"
  "system_het_random_asymmetric"
  "system_het_random_fednova"
  "system_het_partial_C0.5"
  "mu_sensitivity_flower"
  "mu_sweep"
)

# Note: arch_ablation_bn is archived (see its README_PROVENANCE.md) and
# not listed here.

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
for name in "${SWEEPS[@]}"; do
  dir="$RESULTS/$name"
  if is_populated "$dir"; then
    n_files=$(ls "$dir"/test_at_best_*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "→ Analysing $name ($n_files JSON files)..."
    if PYTHONPATH=. python -m fl_dermamnist.analysis.tables \
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
echo "=== Statistical-heterogeneity pipeline (engineered + system-het) ==="
echo ""
if is_populated "$RESULTS/headline" \
   && is_populated "$RESULTS/flower_C0_baseline"; then
  echo "→ Running analyse_statistical_heterogeneity.py..."
  if PYTHONPATH=. python fl_dermamnist/analysis/analyse_statistical_heterogeneity.py \
       >"$RESULTS/thesis_ready_statistical_log.txt" 2>&1; then
    echo "  ✓ wrote results/thesis_ready/data/statistical_heterogeneity_summary.json"
    analysed+=("statistical_heterogeneity_pipeline")
  else
    echo "  ✗ FAILED (see $RESULTS/thesis_ready_statistical_log.txt)"
    failed+=("statistical_heterogeneity_pipeline")
  fi
else
  echo "  – skipped: requires headline/ AND flower_C0_baseline/ populated"
  skipped+=("statistical_heterogeneity_pipeline")
fi

echo ""
echo "=== System-heterogeneity pipeline (S1/S2 + FedNova + partial) ==="
echo ""
if is_populated "$RESULTS/flower_C0_baseline" \
   && is_populated "$RESULTS/system_het_random"; then
  echo "→ Running analyse_system_het.py..."
  if PYTHONPATH=. python fl_dermamnist/results/thesis_ready_system_het/scripts/analyse_system_het.py \
       >"$RESULTS/thesis_ready_system_het_log.txt" 2>&1; then
    echo "  ✓ wrote results/thesis_ready_system_het/data/summary_statistics.json"
    analysed+=("system_het_pipeline")
  else
    echo "  ✗ FAILED (see $RESULTS/thesis_ready_system_het_log.txt)"
    failed+=("system_het_pipeline")
  fi
else
  echo "  – skipped: requires flower_C0_baseline/ AND system_het_random/ populated"
  skipped+=("system_het_pipeline")
fi

echo ""
echo "=== Cross-runtime equivalence check ==="
echo ""
if is_populated "$RESULTS/headline" \
   && is_populated "$RESULTS/flower_C0_baseline"; then
  echo "→ Running compare_equivalence_full_scale..."
  if PYTHONPATH=. python -m fl_dermamnist.experiments.compare_equivalence_full_scale \
       >"$RESULTS/equivalence_full_scale_log.txt" 2>&1; then
    echo "  ✓ wrote results/thesis_ready/data/equivalence_full_scale.json"
    analysed+=("equivalence_full_scale")
  else
    echo "  ✗ FAILED (see $RESULTS/equivalence_full_scale_log.txt)"
    failed+=("equivalence_full_scale")
  fi
else
  echo "  – skipped: requires headline/ AND flower_C0_baseline/ populated"
  skipped+=("equivalence_full_scale")
fi

echo ""
echo "=== Headline extras (sign / Hodges-Lehmann / LOSO / Holm / per-class) ==="
echo ""
if is_populated "$RESULTS/headline"; then
  echo "→ Running analyse_extra_statistics on headline..."
  PYTHONPATH=. python fl_dermamnist/analysis/analyse_extra_statistics.py \
    --results-dir "$RESULTS/headline" \
    >"$RESULTS/headline_extra_stats_log.txt" 2>&1 \
    && echo "  ✓ wrote thesis_ready/data/extra_statistics.json" \
    || echo "  ✗ FAILED (see $RESULTS/headline_extra_stats_log.txt)"
  echo "→ Running plot_per_class_delta on headline..."
  PYTHONPATH=. python fl_dermamnist/figures/plot_per_class_delta.py \
    --results-dir "$RESULTS/headline" \
    >"$RESULTS/headline_per_class_log.txt" 2>&1 \
    && echo "  ✓ wrote thesis_ready/figures/per_class_delta.{png,pdf}" \
    || echo "  ✗ FAILED (see $RESULTS/headline_per_class_log.txt)"
  echo "→ Running analyse_communication_metrics on headline..."
  PYTHONPATH=. python fl_dermamnist/analysis/analyse_communication_metrics.py \
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
echo "  bash infra/local/analyse_all.sh"
echo ""
echo "To inspect overall HPC progress:"
echo "  bash infra/local/check_results.sh"
echo ""
echo "Provenance ledger across all directories:"
echo "  fl_dermamnist/results/PROVENANCE_AUDIT.md"
