#!/bin/bash
#SBATCH -J ray_diag
# FERGUSSON-SL3-GPU: MPHIL-DIS-SL2-GPU ran out of GPU-minutes (2026-06-07).
# Override per run with: sbatch --account=... infra/slurm/ray_diagnostic.sh
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/ray_diag_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/fl_dermamnist/logs/ray_diag_%j.err

# Diagnostic — does Ray work AT ALL on a compute node with the unique-port
# fix? This 5-minute batch job runs the same minimal Ray sanity check the
# interactive Step 2 would run, but doesn't require an interactive shell.
# Submit it and check the log when it finishes; you don't need to babysit.

set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo "============================================================"
echo "Ray-on-HPC diagnostic"
echo "============================================================"
echo "Hostname:     $(hostname)"
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "RAY_TMPDIR:   $RAY_TMPDIR"
echo "/tmp usage:"
df -h /tmp 2>/dev/null || echo "  (df failed)"
echo ""
echo "Venv info:"
which python
python --version
pip show ray 2>/dev/null | grep -E "Name|Version|Location"
echo ""
echo "Checking what's using common Ray ports on this node..."
for port in 6379 6380 6381 8265; do
    result=$(ss -tlnp 2>/dev/null | grep ":$port " | head -1)
    if [ -n "$result" ]; then
        echo "  Port $port: BUSY ($result)"
    else
        echo "  Port $port: free"
    fi
done
echo ""
echo "============================================================"
echo "Attempt 1: Ray init with the production ray_init_args"
echo "(matches what run_one_flower.py uses)"
echo "============================================================"

# This Python block exits with code 0 on success and 1 on failure so
# SLURM's afterok dependency correctly gates downstream jobs.
python <<'PY'
import os, sys, traceback
print(f"Python: {sys.version}")
try:
    import ray
    print(f"Ray version: {ray.__version__}")
except ImportError as e:
    print(f"FATAL: cannot import ray: {e}")
    sys.exit(1)

# Mirror exactly what run_one_flower.py sets in ray_init_args.
ray_init_args = {
    "include_dashboard": False,
    "ignore_reinit_error": True,
    "log_to_driver": False,
}
if "RAY_TMPDIR" in os.environ:
    ray_init_args["_temp_dir"] = os.environ["RAY_TMPDIR"]
slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus and slurm_cpus.isdigit():
    ray_init_args["num_cpus"] = int(slurm_cpus)
ray_init_args["num_gpus"] = 1

print(f"ray_init_args: {ray_init_args}")
print()

try:
    ray.init(**ray_init_args)
    print("✅ SUCCESS: Ray started successfully with production ray_init_args.")
    print(f"   Resources: {ray.cluster_resources()}")
    ray.shutdown()
    print("   Clean shutdown.")
    sys.exit(0)
except Exception as e:
    print(f"❌ FAILED with production args: {type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()
    print()
    # Look at GCS log if any exists.
    import glob
    for log in sorted(glob.glob(f"{os.environ.get('RAY_TMPDIR','')}/session_*/logs/gcs_server.*")):
        print(f"   --- {log} ---")
        try:
            with open(log) as f:
                content = f.read()
                if content:
                    print(content[:2000])
                else:
                    print("   (empty — GCS process died before writing any log)")
        except Exception as ex:
            print(f"   (cannot read: {ex})")
    print()
    print("Trying MINIMAL config (only include_dashboard=False)...")
    try:
        ray.shutdown()  # in case partial state
    except Exception:
        pass
    try:
        ray.init(include_dashboard=False)
        print("✅ MINIMAL works: the issue is one of the extra args (_temp_dir, num_cpus).")
        ray.shutdown()
        sys.exit(2)  # distinguish: minimal works, prod args broken
    except Exception as e2:
        print(f"❌ MINIMAL also failed: {type(e2).__name__}: {e2}")
        traceback.print_exc()
        sys.exit(1)  # Ray fundamentally broken
PY

py_rc=$?
echo ""
echo "============================================================"
echo "Diagnostic exit code: $py_rc"
echo "  0 = ✅ production ray_init_args works — safe to submit experiments"
echo "  1 = ❌ Ray fundamentally broken (venv issue) — do NOT submit"
echo "  2 = ⚠ minimal Ray works but production args fail — code change needed"
echo "============================================================"

# Propagate the python exit code so SLURM's afterok dependency gates
# correctly: only exit 0 (Ray confirmed working) lets dependent jobs run.
exit $py_rc
