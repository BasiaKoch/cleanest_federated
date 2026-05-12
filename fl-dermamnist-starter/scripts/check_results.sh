#!/bin/bash
set -euo pipefail

# Naming convention used by run_experiment.py / config_loader.py:
#   results/{config_stem}_E{local_epochs}_s{seed}/global_test_metrics.json
# All base / mitigation configs inherit local_epochs=5, so the folder is
# results/{stem}_E5_s{seed}/.

DISSERTATION_CORE=(
  fedavg_iid
  fedavg_dir05
  fedprox_dir05
  fedavg_dir01
  fedprox_dir01
  fedavg_pathological
  fedprox_pathological
)

DISSERTATION_MITIGATION=(
  fedprox_dir05_weighted
  fedprox_dir05_focal
  fedavg_dir05_weighted
)

SEEDS=(42 123 456)

missing=0
present=0

check_one() {
  local stem="$1"
  local seed="$2"
  local dir="results/${stem}_E5_s${seed}"
  if [ -f "${dir}/global_test_metrics.json" ]; then
    present=$((present + 1))
    return 0
  else
    missing=$((missing + 1))
    echo "  MISSING: ${dir}"
    return 1
  fi
}

echo "=== Dissertation-focused: core experiments ==="
for stem in "${DISSERTATION_CORE[@]}"; do
  for seed in "${SEEDS[@]}"; do
    check_one "$stem" "$seed" || true
  done
done

echo ""
echo "=== Dissertation-focused: mitigation experiments ==="
for stem in "${DISSERTATION_MITIGATION[@]}"; do
  for seed in "${SEEDS[@]}"; do
    check_one "$stem" "$seed" || true
  done
done

total=$((present + missing))
echo ""
echo "Dissertation set status: ${present}/${total} complete, ${missing} missing"

echo ""
echo "=== All result folders on disk ==="
find results/ -name "global_test_metrics.json" 2>/dev/null | sort || true

echo ""
echo "=== Broader matrix (informational only) ==="
echo "(submit_all_core.sh / submit_mitigation.sh / submit_ablation.sh are"
echo " available manually but are NOT part of the default dissertation run.)"
for config in configs/fedavg_*.yaml configs/fedprox_*.yaml; do
  stem=$(basename "$config" .yaml)
  in_focus=0
  for f in "${DISSERTATION_CORE[@]}"; do
    if [ "$f" = "$stem" ]; then in_focus=1; break; fi
  done
  if [ "$in_focus" = "1" ]; then continue; fi
  for seed in "${SEEDS[@]}"; do
    dir="results/${stem}_E5_s${seed}"
    if [ ! -f "${dir}/global_test_metrics.json" ]; then
      echo "  (extra-matrix missing) ${dir}"
    fi
  done
done
