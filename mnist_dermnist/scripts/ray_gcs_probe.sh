#!/bin/bash
#SBATCH -J ray_probe
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:08:00
#SBATCH --output=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/ray_probe_%j.out
#SBATCH --error=/home/bk489/federated_clean/cleanest_federated/mnist_dermnist/logs/ray_probe_%j.err
#
# Deep GCS-startup probe. The existing ray_diagnostic.sh deletes RAY_TMPDIR on
# exit (so the GCS logs vanish) and globs the wrong session path. This probe:
#   - reports /tmp + /dev/shm free space and key ulimits,
#   - ldd's the gcs_server binary to catch a MISSING SYSTEM LIBRARY (the prime
#     suspect when the same venv worked weeks ago but now fails on every node),
#   - runs a minimal ray.init() and, on failure, dumps gcs_server.{out,err} and
#     raylet.err from the CORRECT path, and copies the session to logs/ before
#     cleanup so the evidence survives.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

echo "=================== ray_gcs_probe ==================="
echo "host:  $(hostname)"
echo "date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job:   ${SLURM_JOB_ID:-none}"
echo ""
echo "== disk: /tmp and /dev/shm =="
df -h /tmp /dev/shm 2>&1
echo ""
echo "== inodes: /tmp =="
df -i /tmp 2>&1
echo ""
echo "== ulimits =="
ulimit -a 2>&1 | grep -Ei "open files|max memory|address space|processes|locked" || ulimit -a
echo ""
echo "== python / ray =="
which python; python --version
python -c "import ray, os; print('ray', ray.__version__); print('ray_dir', os.path.dirname(ray.__file__))"
RAY_DIR=$(python -c "import ray, os; print(os.path.dirname(ray.__file__))")
GCS=$(find "$RAY_DIR" -type f -name gcs_server 2>/dev/null | head -1)
echo "gcs_server binary: ${GCS:-NOT FOUND}"
echo ""
echo "== ldd gcs_server (MISSING libs are the smoking gun) =="
if [ -n "${GCS:-}" ]; then
    ldd "$GCS" 2>&1 | grep -i "not found" && echo "  ^^^ MISSING LIBRARIES ABOVE ^^^" || echo "  (no 'not found' libraries)"
    echo "  --- full ldd (head) ---"; ldd "$GCS" 2>&1 | head -25
else
    echo "  (binary not found — cannot ldd)"
fi
echo ""

PROBE_TMP="/tmp/raygcsprobe-${SLURM_JOB_ID:-$$}"
mkdir -p "$PROBE_TMP"
echo "== minimal ray.init() at $PROBE_TMP (NO cleanup until logs captured) =="
set +e
python - "$PROBE_TMP" <<'PY'
import sys, ray
try:
    ray.init(include_dashboard=False, _temp_dir=sys.argv[1], num_cpus=1)
    print("RAY OK:", ray.cluster_resources())
    ray.shutdown()
except Exception as e:
    print("RAY FAILED:", type(e).__name__, ":", e)
PY
rc=$?
set -e
echo "ray.init exit: $rc"
echo ""
echo "== gcs_server.out =="; tail -40 "$PROBE_TMP"/ray/session_*/logs/gcs_server.out 2>/dev/null || echo "  (none)"
echo "== gcs_server.err =="; tail -40 "$PROBE_TMP"/ray/session_*/logs/gcs_server.err 2>/dev/null || echo "  (none)"
echo "== raylet.err =="; tail -40 "$PROBE_TMP"/ray/session_*/logs/raylet.err 2>/dev/null || echo "  (none)"
echo "== any session log mentioning error/fail/fatal/bind/space =="
grep -rilE "error|fail|fatal|abort|core|no space|bind|cannot|Check failed" \
    "$PROBE_TMP"/ray/session_*/logs/ 2>/dev/null | while read -r f; do
        echo "  -- $f --"; tail -12 "$f"
    done

# Preserve the session for offline inspection, then clean up.
cp -r "$PROBE_TMP"/ray "$REPO_ROOT/mnist_dermnist/logs/ray_probe_session_${SLURM_JOB_ID:-$$}" 2>/dev/null || true
rm -rf "$PROBE_TMP"
echo "=================== end probe (rc=$rc) ==================="
exit "$rc"
