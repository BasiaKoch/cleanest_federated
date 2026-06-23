#!/bin/bash
# Local-epoch (E) sweep on balanced_paired_7_clients — canonical FedProx
# mechanism test on the partition where the FedProx advantage is already
# established (headline Δ macro-F1 ≈ +0.027 at E=20, n=10 paired seeds).
#
# Direct replication-style methodology after Li et al. 2020 MLSys §5.2 /
# Fig. 4: sweep E at fixed federation knobs and show the FedProx-vs-FedAvg
# advantage grows monotonically with E. Mechanism (Karimireddy 2020
# SCAFFOLD §3): client drift accumulates roughly linearly with local SGD
# steps; the proximal term restrains it.
#
# Cost-saver: E=20 is REUSED from the existing 10-seed headline runs in
# fl_dermamnist/results/headline/ via the analyser's --anchor-dir flag,
# so the sweep only needs E ∈ {1, 5, 10, 40} = 4 × 2 × 3 = 24 new jobs.
# (Set MODE=full_plus to also run E=20 fresh in this dir for sanity.)
#
# Usage:
#   PILOT (E=1 only, 2 jobs, ~15 min total):
#       MODE=pilot bash infra/slurm/submit_e_sweep.sh
#
#   FULL  (24 jobs: 4 new E × 2 algos × 3 seeds):
#       MODE=full bash infra/slurm/submit_e_sweep.sh
#
#   FULL+ (30 jobs: also re-run E=20 here for sanity):
#       MODE=full_plus bash infra/slurm/submit_e_sweep.sh
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/bk489/federated_clean/cleanest_federated}
OUT_DIR=fl_dermamnist/results/e_sweep
PARTITION=balanced_paired_7_clients
MU=0.01
# --log-update-norms is essential — half the mechanism evidence comes from
# the per-(round, client) drift trajectories faceted by E.
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
  # Walltime scales with E since per-round local-epoch compute dominates.
  local walltime
  if   [ "$E" -le 5  ]; then walltime="02:00:00"
  elif [ "$E" -le 10 ]; then walltime="04:00:00"
  elif [ "$E" -le 20 ]; then walltime="06:00:00"
  else                       walltime="10:00:00"
  fi
  if ! sbatch \
    --job-name="mn_${algo}_bp_E${E}_mu${mu}_s${seed}" \
    --time="$walltime" \
    "$REPO_ROOT/infra/slurm/slurm_template_flower.sh" \
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
echo "Submitted E-sweep on balanced_paired (MODE=$MODE):"
echo "  partition = $PARTITION"
echo "  seeds     = ${SEEDS[*]}"
echo "  E values  = ${ES[*]}"
echo "  jobs      = $NUM_JOBS  →  $OUT_DIR"
echo ""
echo "When complete, analyse with (uses headline/ as the E=20 anchor):"
echo "  PYTHONPATH=. python fl_dermamnist/analysis/analyse_e_sweep.py"
echo "Monitor:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
