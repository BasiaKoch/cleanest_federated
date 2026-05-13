#!/bin/bash
# All commands for the mnist_dermnist FedAvg/FedProx experiment.
# Run from the repository root: /Users/basiakoch/cleanest_federated
#
# All commands prepend PYTHONPATH=. because tests/scripts use absolute imports.
set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Sanity test — FedProx(μ=0) == FedAvg
# ----------------------------------------------------------------------------
sanity_test() {
  PYTHONPATH=. python -m pytest \
    mnist_dermnist/tests/test_mu_zero_equals_fedavg.py -v
}

# ----------------------------------------------------------------------------
# 2. One FedAvg run: seed=42, E=20
# ----------------------------------------------------------------------------
run_fedavg_one() {
  PYTHONPATH=. python -m mnist_dermnist.experiments.run_one \
    --algorithm fedavg --seed 42 --local-epochs 20
}

# ----------------------------------------------------------------------------
# 3. One FedProx run: μ=0.1, seed=42, E=20
# ----------------------------------------------------------------------------
run_fedprox_one() {
  PYTHONPATH=. python -m mnist_dermnist.experiments.run_one \
    --algorithm fedprox --mu 0.1 --seed 42 --local-epochs 20
}

# ----------------------------------------------------------------------------
# 4. Full headline sweep:
#    10 seeds × {FedAvg, FedProx(μ=0.1)}, E=20, 150 rounds,
#    lr=0.01, batch=32, partition=medical_skew_7_clients
# ----------------------------------------------------------------------------
run_headline_sweep() {
  PYTHONPATH=. python -m mnist_dermnist.experiments.run_headline_sweep \
    --local-epochs 20 \
    --num-rounds 150 \
    --lr 0.01 \
    --batch-size 32 \
    --partition medical_skew_7_clients \
    --out-dir mnist_dermnist/results/headline
  # add --device cuda when on HPC
}

# ----------------------------------------------------------------------------
# 5. Optional E sweep: E ∈ {1, 5, 10, 20, 40} × {FedAvg, FedProx} × 3 seeds
# ----------------------------------------------------------------------------
run_e_sweep() {
  PYTHONPATH=. python -m mnist_dermnist.experiments.run_e_sweep \
    --local-epochs 1 5 10 20 40 \
    --seeds 42 123 456 \
    --num-rounds 100 \
    --partition medical_skew_7_clients \
    --out-dir mnist_dermnist/results/e_sweep
}

# ----------------------------------------------------------------------------
# 6. Analysis from CSV logs
# ----------------------------------------------------------------------------
generate_partition_counts() {
  # Used by the heatmap plot — only needs to run once per (mode, seed)
  PYTHONPATH=. python -m mnist_dermnist.data.partition \
    --mode medical_skew_7_clients --seed 42 \
    --out mnist_dermnist/results/partitions
}

analyze_headline() {
  PYTHONPATH=. python -m mnist_dermnist.analysis.tables \
    --results-dir mnist_dermnist/results/headline --E 20
  PYTHONPATH=. python -m mnist_dermnist.analysis.plots \
    --results-dir mnist_dermnist/results/headline --E 20 \
    --partition-counts mnist_dermnist/results/partitions/partition_medical_skew_7_clients_seed42_counts.csv
}

analyze_e_sweep() {
  for E in 1 5 10 20 40; do
    PYTHONPATH=. python -m mnist_dermnist.analysis.tables \
      --results-dir mnist_dermnist/results/e_sweep --E "$E"
  done
}

# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------
case "${1:-}" in
  sanity)           sanity_test ;;
  fedavg)           run_fedavg_one ;;
  fedprox)          run_fedprox_one ;;
  headline)         run_headline_sweep ;;
  e-sweep)          run_e_sweep ;;
  analyze)          generate_partition_counts && analyze_headline ;;
  analyze-e-sweep)  analyze_e_sweep ;;
  all)
    sanity_test
    generate_partition_counts
    run_headline_sweep
    analyze_headline
    ;;
  *)
    echo "Usage: bash $0 {sanity|fedavg|fedprox|headline|e-sweep|analyze|analyze-e-sweep|all}"
    echo ""
    echo "Steps:"
    echo "  sanity          run μ=0 ≡ FedAvg unit tests"
    echo "  fedavg          one FedAvg run (seed 42, E=20)"
    echo "  fedprox         one FedProx run (μ=0.1, seed 42, E=20)"
    echo "  headline        full sweep (10 seeds × 2 algos × E=20)"
    echo "  e-sweep         optional E sweep (5 E values × 3 seeds × 2 algos)"
    echo "  analyze         partition counts + headline tables + plots"
    echo "  analyze-e-sweep tables only across E values"
    echo "  all             sanity + partition counts + headline + analyze"
    exit 1
    ;;
esac
