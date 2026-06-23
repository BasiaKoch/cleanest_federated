#!/bin/bash
#SBATCH -J ray_probe
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:08:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/ray_probe_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/ray_probe_%j.err
#
# GCS-startup probe v3. Ruled out so far: disk (/tmp,/dev/shm healthy), missing
# libs (ldd clean), FD limit (raised to 1048575 -> still fails), GLIBCXX (lib
# provides 3.4.32 >> needed 3.4.19). Remaining lead: the module-injected
# LD_LIBRARY_PATH (Intel oneAPI MPI / libfabric / UCX / intel-compiler libs)
# conflicts with Ray's bundled gRPC/protobuf at runtime -> GCS dies with an
# empty log. v3 runs ray.init() (a) as-is and (b) with LD_LIBRARY_PATH stripped,
# and confirms torch still sees CUDA without it (torch is a cu128 pip wheel that
# carries its own CUDA libs).
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true   # FD limit (harmless; already ruled out)

echo "=================== ray_gcs_probe v3 ==================="
echo "host: $(hostname)   job: ${SLURM_JOB_ID:-none}   date: $(date -u +%FT%TZ)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "LD_LIBRARY_PATH entries:"; tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | sed 's/^/    /'
echo ""

# Reusable: run ray.init + a torch CUDA check in a subprocess under whatever
# environment the caller sets. Args after the function name run as a prefix.
ray_try() {
    local label="$1"; shift
    local tmp="/tmp/rayp-${SLURM_JOB_ID:-$$}-${label// /_}"
    mkdir -p "$tmp"
    echo "================ ATTEMPT: $label ================"
    set +e
    "$@" python - "$tmp" <<'PY'
import sys
import ray
try:
    ray.init(include_dashboard=False, _temp_dir=sys.argv[1], num_cpus=1)
    print("  RAY OK ->", {k: v for k, v in ray.cluster_resources().items() if k in ("CPU", "GPU")})
    ray.shutdown()
except Exception as e:
    print("  RAY FAILED:", type(e).__name__, ":", str(e).splitlines()[0][:160])
try:
    import torch
    print("  torch", torch.__version__, "cuda.is_available()=", torch.cuda.is_available(),
          "device=", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"))
except Exception as e:
    print("  TORCH CHECK FAILED:", type(e).__name__, ":", str(e)[:160])
PY
    set -e
    rm -rf "$tmp"
    echo ""
}

# (a) current environment (expected: RAY FAILED — reproduces the bug)
ray_try "current-env"

# (b) LD_LIBRARY_PATH stripped (keeps PATH, conda, CUDA_VISIBLE_DEVICES intact)
ray_try "no-LD_LIBRARY_PATH" env -u LD_LIBRARY_PATH

echo "=================== end probe v3 ==================="
echo "Interpretation:"
echo "  - if (b) 'no-LD_LIBRARY_PATH' shows RAY OK + cuda.is_available()=True,"
echo "    the fix is: unset LD_LIBRARY_PATH before launching in the SLURM template."
echo "  - if (b) also fails, paste this log; the issue is deeper than the module env."
