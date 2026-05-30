#!/bin/bash
#SBATCH -J ray_diag
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/ray_diag_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/ray_diag_%j.err

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
echo "Attempt 1: Ray init with all our defensive options"
echo "============================================================"

python <<'PY'
import os, sys, traceback
print(f"Python: {sys.version}")
try:
    import ray
    print(f"Ray version: {ray.__version__}")
except ImportError as e:
    print(f"FATAL: cannot import ray: {e}")
    sys.exit(1)

jid_s = os.environ.get("SLURM_JOB_ID", "999")
jid = int(jid_s) if jid_s.isdigit() else 999
gcs_port = 30000 + (jid % 30000)
print(f"Will try GCS port: {gcs_port}")
print(f"_temp_dir:         {os.environ.get('RAY_TMPDIR', '/tmp/ray')}")
print()

try:
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        _temp_dir=os.environ.get("RAY_TMPDIR", "/tmp/ray"),
        num_cpus=4,
        num_gpus=1,
        port=gcs_port,
        _redis_password=f"flwr_{jid}",
    )
    print("✅ SUCCESS: Ray started with unique port + all defensive options")
    print(f"   Resources: {ray.cluster_resources()}")
    print(f"   Address:   {ray.get_runtime_context().get_node_id()}")
    ray.shutdown()
    print("   Clean shutdown.")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()
    print()
    # Look at GCS log if any exists
    import glob
    for log in sorted(glob.glob(f"{os.environ.get('RAY_TMPDIR','')}/session_*/logs/gcs_server.*")):
        print(f"   --- {log} ---")
        try:
            with open(log) as f:
                content = f.read()
                if content:
                    print(content[:2000])
                else:
                    print("   (empty)")
        except Exception as ex:
            print(f"   (cannot read: {ex})")
PY

py_rc=$?
echo ""
echo "============================================================"
echo "Exit code: $py_rc"
echo "============================================================"

if [ $py_rc -ne 0 ]; then
    echo ""
    echo "Diagnostic FAILED. Attempting fallback: ray.init() with no port specified"
    echo "(let Ray pick a free port itself)..."
    echo ""
    python <<'PY'
import os, ray, traceback
try:
    ray.init(include_dashboard=False, log_to_driver=False,
             _temp_dir=os.environ.get("RAY_TMPDIR", "/tmp/ray"),
             num_cpus=4, num_gpus=1)
    print("✅ FALLBACK WORKED: Ray's auto-port assignment succeeds where port=30000+ fails")
    print(f"   This means the issue is NOT port collision — port=auto would work.")
    ray.shutdown()
except Exception as e:
    print(f"❌ FALLBACK ALSO FAILED: {type(e).__name__}: {e}")
    print(f"   Ray cannot start at all on this compute node — venv or library issue.")
    traceback.print_exc()
PY
fi

exit 0
