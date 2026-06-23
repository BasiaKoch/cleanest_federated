#!/bin/bash
# Local-epoch (E) sweep on Dirichlet α=0.1 — canonical FedProx-mechanism test.
#
# Direct replication of the methodology in Li et al. 2020 MLSys, "Federated
# Optimization in Heterogeneous Networks" (FedProx), §5.2 / Figure 4: sweep
# the number of local epochs E at fixed federation knobs and show that the
# FedProx-vs-FedAvg advantage grows monotonically with E. The mechanism
# (Karimireddy et al. 2020 SCAFFOLD §3) is that client drift accumulates
# approximately linearly with the number of local SGD steps; at E=1 no
# drift accumulates and FedAvg ≈ FedProx, at high E FedAvg may oscillate
# or diverge while FedProx remains stable.
#
# Partition: dirichlet_alpha01_7_clients (Hsu, Qi, Brown 2019; α=0.1 is the
# severe-non-IID NIID-Bench standard from Li, Diao, Chen, He 2022 ICDE).
# Severe but not engineered — defends against the "engineered partition
# favours FedProx" reviewer critique.
#
# Reuses the 10 existing E=20 paired datapoints in
# fl_dermamnist/results/dirichlet_a01/ as the E=20 anchor (zero new cost).
# Seeds {42, 123, 456} match the first three seeds in that directory so
# the E=20 datapoint is bit-comparable to the new E values.
#
# Cost (A100 estimates, 150 rounds, lr=0.01, batch=32):
#   E= 1:  ~8 min  × 6 jobs ≈ 0.8 GPU-h
#   E= 5:  ~25 min × 6 jobs ≈ 2.5 GPU-h
#   E=10:  ~45 min × 6 jobs ≈ 4.5 GPU-h
#   E=40:  ~3 h    × 6 jobs ≈ 18  GPU-h
#   total: ~26 GPU-hours across 24 jobs
#
# Usage:
#   PILOT (E=1 only; both algorithms; one seed):
#       MODE=pilot bash fl_dermamnist/scripts/submit_e_sweep_dirichlet_a01.sh
#
#   FULL  (E∈{1,5,10,40} × 2 algos × 3 seeds = 24 jobs):
#       MODE=full bash fl_dermamnist/scripts/submit_e_sweep_dirichlet_a01.sh
#
#   FULL+ (also re-runs E=20 fresh in this dir instead of reusing dirichlet_a01/):
#       MODE=full_plus bash fl_dermamnist/scripts/submit_e_sweep_dirichlet_a01.sh
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/bk489/federated_clean/cleanest_federated}
MU=0.01
PARTITION=dirichlet_alpha01_7_clients
OUT_DIR=fl_dermamnist/results/e_sweep_dirichlet_a01
# --log-update-norms is essential here — the mechanism plot needs per-(round,
# client) update-norm trajectories to show drift growing with E under FedAvg
# while FedProx restrains it.
EXTRA_ARGS="--log-update-norms"

MODE=${MODE:-pilot}
case "$MODE" in
  pilot)      SEEDS=(42)              ES=(1) ;;
  full)       SEEDS=(42 123 456)      ES=(1 5 10 40) ;;
  full_plus)  SEEDS=(42 123 456)      ES=(1 5 10 20 40) ;;
  *)          echo "Unknown MODE=$MODE (use pilot|full|full_plus)" >&2; exit 1 ;;
esac

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3" E="$4"
  # Allow ~3h per E=40 job; cap at 10h SLURM walltime for safety margin.
  local walltime
  if   [ "$E" -le 5  ]; then walltime="02:00:00"
  elif [ "$E" -le 10 ]; then walltime="04:00:00"
  elif [ "$E" -le 20 ]; then walltime="06:00:00"
  else                       walltime="10:00:00"
  fi
  if ! sbatch \
    --job-name="mn_${algo}_dirA01_E${E}_mu${mu}_s${seed}" \
    --time="$walltime" \
    "$REPO_ROOT/fl_dermamnist/scripts/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$E" "$OUT_DIR" "$PARTITION" "$EXTRA_ARGS"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed E=$E"
    FAILED+=("$algo $mu $seed E=$E")
  fi
  sleep 3
}

for E in "${ES[@]}"; do
  for s in "${SEEDS[@]}"; do
    submit fedavg  0.0  "$s" "$E"
    submit fedprox "$MU" "$s" "$E"
  done
done

NUM_JOBS=$(( ${#SEEDS[@]} * ${#ES[@]} * 2 ))
echo ""
echo "Submitted E-sweep on Dirichlet α=0.1 (MODE=$MODE):"
echo "  partition = $PARTITION"
echo "  seeds     = ${SEEDS[*]}"
echo "  E values  = ${ES[*]}"
echo "  jobs      = $NUM_JOBS  →  $OUT_DIR"
echo ""
echo "When complete, analyse with:"
echo "  PYTHONPATH=. python fl_dermamnist/scripts/analyse_e_sweep_dirichlet_a01.py"
echo "Monitor:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
