#!/bin/bash
# Small-hospital case-study baselines — submits all eight jobs to SLURM
# with proper dependency ordering. Each job is its own sbatch
# allocation; the fine-tuning jobs (which depend on the federated
# checkpoints) use --dependency=afterok so they only launch after the
# corresponding checkpoint-save job has completed successfully.
#
# Job graph (8 jobs total, ~5-6 GPU-hours of total compute):
#
#   FedAvg checkpoint  ─┐
#                       ├──> Fine-tune  FedAvg + Client 0
#                       └──> Fine-tune  FedAvg + Client 1
#   FedProx checkpoint ─┐
#                       ├──> Fine-tune  FedProx + Client 0
#                       └──> Fine-tune  FedProx + Client 1
#   Local-only Client 0  (no dependency)
#   Local-only Client 1  (no dependency)
#
# Idempotency
# -----------
# If a federated checkpoint .pt file already exists for seed 42, the
# corresponding checkpoint-save job is skipped and the fine-tuning jobs
# launch immediately (no dependency). This makes the script safe to
# re-run after a partial failure --- finished work is not re-done.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

SEED=42
PARTITION=two_client_90_10_rare_stress
LOCAL_EPOCHS=20
NUM_ROUNDS=150
MU_PROX=0.01

FED_DIR=mnist_dermnist/results/two_client_90_10_rare_stress
LOCAL_OUT=mnist_dermnist/results/small_hospital_local_only
FT_OUT=mnist_dermnist/results/small_hospital_finetune
mkdir -p "$LOCAL_OUT" "$FT_OUT" "$FED_DIR" mnist_dermnist/logs

FLOWER_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh"
SP_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_single_process.sh"

# All informational messages from helpers go to stderr; only the SLURM
# job id is printed to stdout. This is critical because callers use
# `JID=$(submit_*)` to capture the id, and any extra stdout becomes part
# of the captured string. (Lesson learned the hard way from the first
# revision of this script.)
log()  { echo "$@" >&2; }

# Submit a federated run with --save-best-checkpoint. Prints ONLY the
# jobid to stdout. Returns non-zero (and prints nothing to stdout) on
# failure.
submit_federated_checkpoint() {
    local algo="$1" mu="$2" name="$3"
    local jid err
    err=$(mktemp)
    if ! jid=$(sbatch --parsable \
        --job-name="$name" \
        "$FLOWER_TPL" \
        "$algo" "$mu" "$SEED" "$LOCAL_EPOCHS" \
        "$FED_DIR" "$PARTITION" \
        "--num-rounds $NUM_ROUNDS --save-best-checkpoint --log-update-norms" \
        2>"$err"); then
        log "  FAILED sbatch for $name: $(cat "$err")"
        rm -f "$err"
        return 1
    fi
    rm -f "$err"
    echo "$jid"
}

# Submit a single-process job. Optional 4th argument is a colon-
# separated list of dependency jobids; if non-empty, the job will only
# start after all listed jobs exit cleanly.
submit_single_process() {
    local name="$1" module="$2" args="$3" deps="${4:-}"
    local dep_arg=()
    if [ -n "$deps" ]; then
        dep_arg=(--dependency=afterok:"$deps")
    fi
    local jid err
    err=$(mktemp)
    if ! jid=$(sbatch --parsable \
        --job-name="$name" \
        "${dep_arg[@]}" \
        "$SP_TPL" \
        "$module" "$args" \
        2>"$err"); then
        log "  FAILED sbatch for $name: $(cat "$err")"
        rm -f "$err"
        return 1
    fi
    rm -f "$err"
    echo "$jid"
}

# Track results in parent shell (NOT in subshells).
declare -a JOBS_SUMMARY=()
declare -a FAILED=()

record() {
    # record <jid> <name>; jid may be empty if submission failed.
    if [ -n "${1:-}" ]; then
        JOBS_SUMMARY+=("$1  $2")
        log "  $2  ->  SLURM job $1"
    else
        FAILED+=("$2")
        log "  $2  ->  FAILED (see log above)"
    fi
}

CKPT_FA="$FED_DIR/best_state_fedavg_mu0.0_E${LOCAL_EPOCHS}_s${SEED}.pt"
CKPT_FP="$FED_DIR/best_state_fedprox_mu${MU_PROX}_E${LOCAL_EPOCHS}_s${SEED}.pt"

log "============================================================"
log "Step 1: federated jobs (re-run with --save-best-checkpoint)"
log "============================================================"

FA_JID=""
FP_JID=""

if [ -f "$CKPT_FA" ]; then
    log "  $CKPT_FA already exists; skipping FedAvg checkpoint job."
else
    FA_JID=$(submit_federated_checkpoint fedavg  0.0 "mn_sh_ckpt_fedavg_s${SEED}") || true
    record "$FA_JID" "mn_sh_ckpt_fedavg_s${SEED}"
fi

if [ -f "$CKPT_FP" ]; then
    log "  $CKPT_FP already exists; skipping FedProx checkpoint job."
else
    FP_JID=$(submit_federated_checkpoint fedprox $MU_PROX "mn_sh_ckpt_fedprox_s${SEED}") || true
    record "$FP_JID" "mn_sh_ckpt_fedprox_s${SEED}"
fi

log ""
log "============================================================"
log "Step 2a: local-only baselines (no dependency)"
log "============================================================"
LOCAL0_ARGS="--seed $SEED --partition $PARTITION --client-id 0 \
--num-epochs 3000 --eval-every 20 --device cuda --out-dir $LOCAL_OUT"
LOCAL1_ARGS="--seed $SEED --partition $PARTITION --client-id 1 \
--num-epochs 3000 --eval-every 20 --device cuda --out-dir $LOCAL_OUT"

JID=$(submit_single_process \
    "mn_sh_local_only_c0_s${SEED}" \
    "mnist_dermnist.experiments.run_local_only" \
    "$LOCAL0_ARGS" "") || true
record "$JID" "mn_sh_local_only_c0_s${SEED}"

JID=$(submit_single_process \
    "mn_sh_local_only_c1_s${SEED}" \
    "mnist_dermnist.experiments.run_local_only" \
    "$LOCAL1_ARGS" "") || true
record "$JID" "mn_sh_local_only_c1_s${SEED}"

log ""
log "============================================================"
log "Step 2b: fine-tuning baselines (depend on Step 1 checkpoints)"
log "============================================================"

for cid in 0 1; do
    # FedAvg + per-client fine-tuning. Depends on FA_JID if we just
    # submitted the FedAvg checkpoint job; otherwise no dependency
    # (checkpoint already on disk).
    FT_FA_ARGS="--checkpoint $CKPT_FA --seed $SEED --partition $PARTITION \
--client-id $cid --num-epochs 5 --lr 0.001 --device cuda --out-dir $FT_OUT"
    JID=$(submit_single_process \
        "mn_sh_ft_fedavg_c${cid}_s${SEED}" \
        "mnist_dermnist.experiments.run_finetune" \
        "$FT_FA_ARGS" \
        "$FA_JID") || true
    record "$JID" "mn_sh_ft_fedavg_c${cid}_s${SEED}"

    # FedProx + per-client fine-tuning.
    FT_FP_ARGS="--checkpoint $CKPT_FP --seed $SEED --partition $PARTITION \
--client-id $cid --num-epochs 5 --lr 0.001 --device cuda --out-dir $FT_OUT"
    JID=$(submit_single_process \
        "mn_sh_ft_fedprox_c${cid}_s${SEED}" \
        "mnist_dermnist.experiments.run_finetune" \
        "$FT_FP_ARGS" \
        "$FP_JID") || true
    record "$JID" "mn_sh_ft_fedprox_c${cid}_s${SEED}"
done

echo ""
echo "============================================================"
echo "Submitted ${#JOBS_SUMMARY[@]} jobs (${#FAILED[@]} failed)."
echo "============================================================"
for entry in "${JOBS_SUMMARY[@]}"; do
    echo "  $entry"
done
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs at:      $REPO_ROOT/mnist_dermnist/logs/<job-name>_<jobid>.{out,err}"
echo ""
echo "Outputs land in:"
echo "  - Federated checkpoints: $FED_DIR/best_state_*.pt"
echo "  - Local-only baselines:  $LOCAL_OUT/"
echo "  - Fine-tuning baselines: $FT_OUT/"
echo ""
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "WARNING: ${#FAILED[@]} submissions failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
