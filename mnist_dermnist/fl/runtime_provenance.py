"""Runtime provenance fields stamped into every result JSON.

Imported by every entry point that writes a ``test_at_best_*.json`` file
so each output carries enough context to reconstruct the run later:

  ``git_commit``       SHA of HEAD at job start (or None on git failure)
  ``git_dirty``        True if the working tree had uncommitted changes
  ``hostname``         The HPC node / local machine the job ran on
  ``python_version``   Python interpreter version
  ``torch_version``    PyTorch version
  ``cuda_available``   Whether CUDA was visible to torch
  ``device_name``      The first CUDA device's name (or None on CPU)
  ``run_started_at``   ISO-8601, timezone-aware; caller-supplied
  ``run_finished_at``  ISO-8601, timezone-aware; defaults to "now"

Defensive design: every git call has its own try/except, so a job that
runs from a tarball with no .git/ directory does not crash inside the
provenance helper — it just records ``git_commit=None``, ``git_dirty=None``.
The HPC environment may or may not have git available; the runner must
not depend on it.
"""
from __future__ import annotations

import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_rev_parse_head() -> str | None:
    """Return git HEAD SHA, or None if git is unavailable / fails."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        return None


def _git_is_dirty() -> bool | None:
    """Return True if the working tree has uncommitted modifications.

    Returns None on git failure (rather than False) so the JSON
    consumer can distinguish "git said clean" from "git couldn't tell".
    """
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return bool(output.strip())
    except Exception:
        return None


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 timezone-aware string."""
    return datetime.now(timezone.utc).isoformat()


def collect_runtime_provenance(
    run_started_at: str,
    run_finished_at: str | None = None,
) -> dict[str, Any]:
    """Return the runtime provenance dict to merge into a result JSON.

    Parameters
    ----------
    run_started_at : str
        ISO-8601 timestamp captured at the start of the run (caller's
        responsibility — typically the first line of ``main()``).
    run_finished_at : str, optional
        ISO-8601 timestamp at end of run. If None, captured here via
        :func:`utc_now_iso`.
    """
    # Local import keeps this module light at import time and avoids a
    # torch dependency for code paths that don't need it (none of the
    # callers actually exercise that, but it costs nothing).
    import torch

    cuda_available = torch.cuda.is_available()
    device_name: str | None = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = None

    return {
        "git_commit": _git_rev_parse_head(),
        "git_dirty": _git_is_dirty(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at or utc_now_iso(),
    }
