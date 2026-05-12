#!/bin/bash
set -euo pipefail

echo "=== Completed experiments ==="
find results/ -name "global_test_metrics.json" | sort || true

echo ""
echo "=== Expected but missing ==="
for config in configs/fedavg_*.yaml configs/fedprox_*.yaml; do
  stem=$(basename "$config" .yaml)
  for seed in 42 123 456; do
    dir="results/${stem}_E5_s${seed}"
    if [ ! -f "${dir}/global_test_metrics.json" ]; then
      echo "MISSING: ${dir}"
    fi
  done
done
