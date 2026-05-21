#!/bin/bash
# pre_submission_check.sh — verify everything is ready for HPC submission.
#
# Runs the same checks both locally (before committing to a long queue)
# and on the HPC login node (before sbatch). Exits 0 only if every check
# passes. Exit non-zero with a clear PASS/FAIL log otherwise.
#
# Usage:
#   bash mnist_dermnist/scripts/pre_submission_check.sh
#
# What this checks
# ----------------
#   1. The dataset npz exists at the expected path.
#   2. The python interpreter in PATH is the venv's (or, on HPC, the
#      sourced venv's). Reports python version.
#   3. flwr is importable and at the pinned version (requirements.txt).
#   4. torch is importable and at the pinned version.
#   5. CUDA visibility is reported (informational; not a failure if CPU).
#   6. All 13 expected submit/template scripts exist + pass bash -n.
#   7. All federated submit_*.sh scripts route through a Flower template
#      (mechanical Flower-only-on-HPC rule, mirrored from the pytest
#      guard).
#   8. The pytest suite passes.
#
# Each step prints "PASS:" or "FAIL:" and the script's exit code is the
# count of failed steps. Zero = ready to submit.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

FAIL_COUNT=0
declare -a FAILURES

step() {
    local name="$1"
    local cmd="$2"
    printf "  [%-32s] " "$name"
    if eval "$cmd" >/tmp/preflight_step.log 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILURES+=("$name")
        sed 's/^/        /' /tmp/preflight_step.log | head -8
    fi
}

echo "===================================================================="
echo " PRE-SUBMISSION CHECK   $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo " REPO_ROOT=$REPO_ROOT"
echo "===================================================================="

# 1. Dataset
step "dataset_present" \
    'test -f "$REPO_ROOT/dermamnist_64.npz"'

# 2. Python interpreter (informational)
step "python_in_venv" \
    'which python | grep -q "/.venv/" && python --version'

# 3. flwr import + pinned version (best-effort: warn but do not fail on mismatch)
EXPECTED_FLWR=$(grep '^flwr==' "$REPO_ROOT/requirements.txt" | cut -d= -f3)
step "flwr_importable_and_pinned" \
    "python -c \"import flwr,sys; assert flwr.__version__=='${EXPECTED_FLWR}', f'flwr {flwr.__version__} != ${EXPECTED_FLWR}'; print(flwr.__version__)\""

# 4. torch import + pinned version
EXPECTED_TORCH=$(grep '^torch==' "$REPO_ROOT/requirements.txt" | cut -d= -f3)
step "torch_importable_and_pinned" \
    "python -c \"import torch,sys; assert torch.__version__=='${EXPECTED_TORCH}', f'torch {torch.__version__} != ${EXPECTED_TORCH}'; print(torch.__version__)\""

# 5. CUDA (informational only; does not fail if absent — local checks
# typically run on CPU machines)
echo -n "  [cuda_visibility               ] "
python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'cpu-only')" 2>/dev/null || echo "(torch import failed; see above)"

# 6. Shell scripts: existence + bash -n
SUBMIT_SCRIPTS=( $(ls "$REPO_ROOT/mnist_dermnist/scripts/"submit_*.sh "$REPO_ROOT/mnist_dermnist/scripts/"slurm_template*.sh "$REPO_ROOT/mnist_dermnist/scripts/"slurm_centralised.sh 2>/dev/null) )
step "shell_script_syntax_valid" \
    'for f in "${SUBMIT_SCRIPTS[@]}"; do bash -n "$f" || exit 1; done'

# 7. Flower-only-on-HPC mechanical rule (via pytest)
step "flower_only_hpc_guard_test" \
    'PYTHONPATH=. python -m pytest mnist_dermnist/tests/test_no_pure_pytorch_hpc_submission.py -q'

# 8. Full pytest suite
step "pytest_full_suite" \
    'PYTHONPATH=. python -m pytest mnist_dermnist/tests/ -q'

echo "===================================================================="
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo " ALL CHECKS PASSED. Ready for HPC submission."
    echo "===================================================================="
    exit 0
else
    echo " ${FAIL_COUNT} CHECK(S) FAILED:"
    for f in "${FAILURES[@]}"; do echo "   - $f"; done
    echo " Do NOT submit until every check passes."
    echo "===================================================================="
    exit "$FAIL_COUNT"
fi
