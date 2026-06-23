#!/bin/bash
# Engineered 2-client 90/10 rare-class stress test — FedAvg vs FedProx.
#
# Deliberately stressful partition designed to expose FedAvg's failure
# mode under combined quantity- AND label-skew, the regime where the
# FedProx proximal term has the largest theoretical room to help.
# Client 0 (~86%) holds every COMMON class in full — nevi (5), benign
# keratosis (2), actinic (0), basal (1) — and ZERO of the three
# critical/rare classes melanoma (4), dermatofibroma (3), vascular (6).
# Client 1 (~14%) holds 100% of melanoma + dermato + vascular and
# nothing else. The class-disjoint design maximises the sample-count-
# weighted FedAvg bias toward Client 0's "no-melanoma" local objective.
#
# Same federation knobs as the headline (E=20 local epochs, 150 rounds,
# lr=0.01, batch=32, C=1.0, μ=0.01 for FedProx) so the contrast against
# the existing results/headline/ and results/dirichlet_a01/ runs is
# clean.
#
# Usage:
#   PILOT (seed 42 only, FedAvg + FedProx with μ=0.01):
#       MODE=pilot bash infra/slurm/submit_two_client_90_10_rare_stress.sh
#
#   PILOT-plus (seed 42 + one extra μ for cheap μ sensitivity):
#       MODE=pilot_mu bash infra/slurm/submit_two_client_90_10_rare_stress.sh
#
#   FULL (3 seeds × {FedAvg, FedProx} = 6 jobs):
#       MODE=full bash infra/slurm/submit_two_client_90_10_rare_stress.sh
#
#   FULL5 (5 seeds × {FedAvg, FedProx} = 10 jobs):
#       MODE=full5 bash infra/slurm/submit_two_client_90_10_rare_stress.sh
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/bk489/federated_clean/cleanest_federated}
LOCAL_EPOCHS=20
MU=0.01
PARTITION=two_client_90_10_rare_stress
OUT_DIR=fl_dermamnist/results/two_client_90_10_rare_stress
# --log-update-norms emits per-(round, client) client_update_norms_*.csv
# beside the JSON — the diagnostic the supervisor asked for under stress.
EXTRA_ARGS="--log-update-norms"

MODE=${MODE:-pilot}
case "$MODE" in
  pilot)     SEEDS=(42)                      MUS=("$MU") ;;
  pilot_mu)  SEEDS=(42)                      MUS=("$MU" "0.1") ;;
  full)      SEEDS=(42 123 456)              MUS=("$MU") ;;
  full5)     SEEDS=(42 123 456 789 999)      MUS=("$MU") ;;
  *)         echo "Unknown MODE=$MODE (use pilot|pilot_mu|full|full5)" >&2; exit 1 ;;
esac

mkdir -p "$REPO_ROOT/$OUT_DIR" "$REPO_ROOT/fl_dermamnist/logs"

FAILED=()
submit() {
  local algo="$1" mu="$2" seed="$3"
  if ! sbatch \
    --job-name="mn_${algo}_2c9010_mu${mu}_E${LOCAL_EPOCHS}_s${seed}" \
    "$REPO_ROOT/infra/slurm/slurm_template_flower.sh" \
    "$algo" "$mu" "$seed" "$LOCAL_EPOCHS" "$OUT_DIR" "$PARTITION" "$EXTRA_ARGS"; then
    echo "  FAILED to submit: algo=$algo mu=$mu seed=$seed"
    FAILED+=("$algo $mu $seed")
  fi
  sleep 3
}

# FedAvg always (μ=0). FedProx for each requested μ.
for s in "${SEEDS[@]}"; do
  submit fedavg 0.0 "$s"
  for m in "${MUS[@]}"; do
    submit fedprox "$m" "$s"
  done
done

NUM_PROX=$(( ${#SEEDS[@]} * ${#MUS[@]} ))
NUM_AVG=${#SEEDS[@]}
TOTAL=$(( NUM_AVG + NUM_PROX ))
echo ""
echo "Submitted engineered 2-client stress sweep (MODE=$MODE):"
echo "  partition  = $PARTITION"
echo "  seeds      = ${SEEDS[*]}"
echo "  μ values   = ${MUS[*]}  (FedProx)  +  FedAvg always"
echo "  jobs       = $TOTAL  ($NUM_AVG FedAvg + $NUM_PROX FedProx) → $OUT_DIR"
echo ""
echo "When complete, analyse with:"
echo "  PYTHONPATH=. python fl_dermamnist/analysis/analyse_two_client_90_10_rare_stress.py"
echo "Monitor:  squeue -u \$USER"

if [ ${#FAILED[@]} -ne 0 ]; then
  echo ""
  echo "WARNING: ${#FAILED[@]} submissions failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
