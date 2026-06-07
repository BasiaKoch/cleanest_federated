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
# Deep GCS-startup probe v2. v1 confirmed: /tmp & /dev/shm healthy, no missing
# libs, ray 2.31.0, ulimit -n = 1024 (LOW). v2 tests the two remaining
# hypotheses and CAPTURES the real gcs_server error (v1 globbed the wrong
# session path — with _temp_dir Ray uses <tmp>/session_*, no ray/ level — and
# deleted it):
#   (1) FD limit: raise `ulimit -n` and retry ray.init().
#   (2) libstdc++/env: report LD_LIBRARY_PATH / CONDA + GLIBCXX provided-vs-needed.
# The session dir is copied to logs/ BEFORE cleanup so gcs_server.err survives.
set -uo pipefail

REPO_ROOT=/home/bk489/federated_clean/cleanest_federated
VENV_DIR=/home/bk489/federated_clean/.venv
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

echo "=================== ray_gcs_probe v2 ==================="
echo "host: $(hostname)   job: ${SLURM_JOB_ID:-none}   date: $(date -u +%FT%TZ)"
echo ""
echo "== FD limits BEFORE =="
echo "  soft nofile: $(ulimit -Sn)   hard nofile: $(ulimit -Hn)"
echo "  /proc/sys/fs/file-max: $(cat /proc/sys/fs/file-max 2>/dev/null)"
# (1) Raise the soft open-files limit as high as the hard limit allows.
NEWLIM=$(ulimit -Hn)
[ "$NEWLIM" = "unlimited" ] && NEWLIM=1048576
ulimit -n "$NEWLIM" 2>/dev/null || true
echo "== FD limits AFTER raise =="
echo "  soft nofile: $(ulimit -Sn)"
echo ""
echo "== environment (lib-conflict hypothesis) =="
echo "  CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-unset}"
echo "  CONDA_PREFIX:      ${CONDA_PREFIX:-unset}"
echo "  LD_LIBRARY_PATH:   ${LD_LIBRARY_PATH:-unset}"
echo "  which python:      $(which python)"
echo ""
echo "== GLIBCXX: needed by gcs_server vs provided by the resolved libstdc++ =="
GCS="$VENV_DIR/lib/python3.9/site-packages/ray/core/src/ray/gcs/gcs_server"
LIBSTDCXX=$(ldd "$GCS" 2>/dev/null | awk '/libstdc\+\+/{print $3}')
echo "  resolved libstdc++: $LIBSTDCXX"
echo "  gcs_server needs (max): $(strings -a "$GCS" 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -3 | tr '\n' ' ')"
echo "  libstdc++ provides (max): $(strings -a "$LIBSTDCXX" 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -3 | tr '\n' ' ')"
echo ""

PROBE_TMP="/tmp/raygcsprobe-${SLURM_JOB_ID:-$$}"
mkdir -p "$PROBE_TMP"
echo "== ray.init() AFTER raising FD limit, temp_dir=$PROBE_TMP =="
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
echo "ray.init python exit: $rc"
echo ""
# Copy the WHOLE temp tree to logs/ before anything is removed (handles both
# <tmp>/session_* and <tmp>/ray/session_* layouts).
DEST="$REPO_ROOT/mnist_dermnist/logs/ray_probe_session_${SLURM_JOB_ID:-$$}"
mkdir -p "$DEST"
cp -r "$PROBE_TMP"/. "$DEST"/ 2>/dev/null || true
echo "== gcs_server.err (correct path) =="
find "$PROBE_TMP" -path '*logs/gcs_server.err' -exec tail -40 {} \; 2>/dev/null || echo "  (none)"
echo "== gcs_server.out =="
find "$PROBE_TMP" -path '*logs/gcs_server.out' -exec tail -40 {} \; 2>/dev/null || echo "  (none)"
echo "== raylet.err =="
find "$PROBE_TMP" -path '*logs/raylet.err' -exec tail -40 {} \; 2>/dev/null || echo "  (none)"
echo "== any session log mentioning error/fail/fatal/bind/Check failed =="
find "$PROBE_TMP" -path '*logs/*' -type f 2>/dev/null | while read -r f; do
    if grep -qiE "error|fail|fatal|abort|core|no space|bind|cannot|Check failed|GLIBCXX|symbol" "$f" 2>/dev/null; then
        echo "  -- $f --"; grep -iE "error|fail|fatal|abort|core|no space|bind|cannot|Check failed|GLIBCXX|symbol" "$f" | tail -12
    fi
done
rm -rf "$PROBE_TMP"
echo "=================== end probe v2 (rc=$rc); session copied to $DEST ==================="
exit "$rc"
