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
#   (re-runs seed 42    │
#    with --save-best-  │
#    checkpoint flag)   │
#                       ├──> Fine-tune  FedAvg + Client 0
#                       └──> Fine-tune  FedAvg + Client 1
#
#   FedProx checkpoint ─┐
#   (re-runs seed 42    │
#    with --save-best-  │
#    checkpoint flag)   │
#                       ├──> Fine-tune  FedProx + Client 0
#                       └──> Fine-tune  FedProx + Client 1
#
#   Local-only Client 0  (no dependency)
#   Local-only Client 1  (no dependency)
#
# References for the baselines this implements
# --------------------------------------------
#   - Sheller et al. (2018, 2020), Roth et al. (2020), Pati et al. (2022):
#     local-only-vs-federated comparison.
#   - Yu et al. (2022): local fine-tuning protocol (5 epochs, lr=0.001).
#   - Collins et al. (2021) FedRep; Marfoq et al. (2021): personalised-FL
#     framing for the per-client gap discussion.
#
# Usage
# -----
#   bash mnist_dermnist/scripts/submit_small_hospital_baselines.sh
#
#   The script prints the eight resulting SLURM job ids; monitor with
#       squeue -u $USER
#   and once all complete, the analysis script can ingest both output
#   directories.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
cd "$REPO_ROOT"

SEED=42
PARTITION=two_client_90_10_rare_stress
LOCAL_EPOCHS=20    # Federation E_max; matches existing seed-42 runs.
NUM_ROUNDS=150     # Federation R; matches existing seed-42 runs.
MU_PROX=0.01

FED_DIR=mnist_dermnist/results/two_client_90_10_rare_stress
LOCAL_OUT=mnist_dermnist/results/small_hospital_local_only
FT_OUT=mnist_dermnist/results/small_hospital_finetune
mkdir -p "$LOCAL_OUT" "$FT_OUT" "$FED_DIR" mnist_dermnist/logs

FLOWER_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_flower.sh"
SP_TPL="$REPO_ROOT/mnist_dermnist/scripts/slurm_template_single_process.sh"

FAILED=()
JOBS=()

# Submit a federated run with --save-best-checkpoint enabled. Returns
# the SLURM job id on stdout (last token of the `Submitted batch job N`
# line from sbatch).
submit_federated_checkpoint() {
    local algo="$1" mu="$2" name="$3"
    local jid
    if ! jid=$(sbatch --parsable \
        --job-name="$name" \
        "$FLOWER_TPL" \
        "$algo" "$mu" "$SEED" "$LOCAL_EPOCHS" \
        "$FED_DIR" "$PARTITION" \
        "--num-rounds $NUM_ROUNDS --save-best-checkpoint --log-update-norms"); then
        echo "  FAILED to submit federated checkpoint job: $name" >&2
        FAILED+=("$name")
        return 1
    fi
    echo "  $name -> SLURM job $jid"
    JOBS+=("$jid:$name")
    echo "$jid"
}

# Submit a single-process job. If $deps is non-empty, adds
# --dependency=afterok:<jid>[:<jid>...].
submit_single_process() {
    local name="$1" module="$2" args="$3" deps="$4"
    local dep_arg=""
    if [ -n "$deps" ]; then
        dep_arg="--dependency=afterok:$deps"
    fi
    local jid
    if ! jid=$(sbatch --parsable \
        --job-name="$name" \
        $dep_arg \
        "$SP_TPL" \
        "$module" "$args"); then
        echo "  FAILED to submit single-process job: $name" >&2
        FAILED+=("$name")
        return 1
    fi
    echo "  $name -> SLURM job $jid"
    JOBS+=("$jid:$name")
    echo "$jid"
}

echo "============================================================"
echo "Step 1: re-run federated jobs to capture best-val checkpoints"
echo "============================================================"
# These two jobs re-run seed 42 of FedAvg and FedProx on the 90/10
# partition, this time saving the best-val state_dict to a .pt file.
FA_JID=$(submit_federated_checkpoint fedavg  0.0       "mn_sh_ckpt_fedavg_s${SEED}")
FP_JID=$(submit_federated_checkpoint fedprox $MU_PROX  "mn_sh_ckpt_fedprox_s${SEED}")

echo ""
echo "============================================================"
echo "Step 2a: local-only baselines (no dependency, run immediately)"
echo "============================================================"
LOCAL0_ARGS="--seed $SEED --partition $PARTITION --client-id 0 \
--num-epochs 3000 --eval-every 20 --device cuda --out-dir $LOCAL_OUT"
LOCAL1_ARGS="--seed $SEED --partition $PARTITION --client-id 1 \
--num-epochs 3000 --eval-every 20 --device cuda --out-dir $LOCAL_OUT"

submit_single_process \
    "mn_sh_local_only_c0_s${SEED}" \
    "mnist_dermnist.experiments.run_local_only" \
    "$LOCAL0_ARGS" \
    "" > /dev/null

submit_single_process \
    "mn_sh_local_only_c1_s${SEED}" \
    "mnist_dermnist.experiments.run_local_only" \
    "$LOCAL1_ARGS" \
    "" > /dev/null

echo ""
echo "============================================================"
echo "Step 2b: fine-tuning baselines (depend on Step 1 checkpoints)"
echo "============================================================"
CKPT_FA="$FED_DIR/best_state_fedavg_mu0.0_E${LOCAL_EPOCHS}_s${SEED}.pt"
CKPT_FP="$FED_DIR/best_state_fedprox_mu${MU_PROX}_E${LOCAL_EPOCHS}_s${SEED}.pt"

for cid in 0 1; do
    # FedAvg + per-client fine-tuning, depends on the FedAvg checkpoint.
    FT_FA_ARGS="--checkpoint $CKPT_FA --seed $SEED --partition $PARTITION \
--client-id $cid --num-epochs 5 --lr 0.001 --device cuda --out-dir $FT_OUT"
    submit_single_process \
        "mn_sh_ft_fedavg_c${cid}_s${SEED}" \
        "mnist_dermnist.experiments.run_finetune" \
        "$FT_FA_ARGS" \
        "$FA_JID" > /dev/null

    # FedProx + per-client fine-tuning, depends on the FedProx checkpoint.
    FT_FP_ARGS="--checkpoint $CKPT_FP --seed $SEED --partition $PARTITION \
--client-id $cid --num-epochs 5 --lr 0.001 --device cuda --out-dir $FT_OUT"
    submit_single_process \
        "mn_sh_ft_fedprox_c${cid}_s${SEED}" \
        "mnist_dermnist.experiments.run_finetune" \
        "$FT_FP_ARGS" \
        "$FP_JID" > /dev/null
done

echo ""
echo "============================================================"
echo "All ${#JOBS[@]} jobs submitted."
echo "============================================================"
for entry in "${JOBS[@]}"; do
    echo "  ${entry%%:*}  ${entry#*:}"
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
