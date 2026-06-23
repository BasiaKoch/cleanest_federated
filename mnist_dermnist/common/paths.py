"""Repository path resolution for the DermaMNIST FL project.

Single source of truth for where the repo, package, results, and thesis-ready
artefacts live, so scripts do not depend on fragile
``Path(__file__).parents[...]`` arithmetic (which silently breaks when a script
is moved to a different directory depth).

Repo-root resolution order:
  1. ``$FED_REPO_ROOT`` if set;
  2. otherwise this file's own location — ``mnist_dermnist/common/paths.py``,
     i.e. two parents up is the repository root. Because this module lives at a
     fixed location inside the package, the result is independent of where any
     *caller* script sits.

The results root can be overridden independently via ``$FED_RESULTS_ROOT``
(default: ``<repo>/mnist_dermnist/results``).

Standard library only; no third-party dependencies.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "package_root",
    "results_root",
    "thesis_ready_root",
    "thesis_data_dir",
    "thesis_figures_dir",
]

_PACKAGE_NAME = "mnist_dermnist"


def _resolved(p: "str | os.PathLike[str]") -> Path:
    return Path(p).expanduser().resolve()


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            f"Run from inside the repository, or set FED_REPO_ROOT / "
            f"FED_RESULTS_ROOT to point at it."
        )
    return path


def repo_root() -> Path:
    """Repository root. Honours ``$FED_REPO_ROOT``; else derived from this file."""
    env = os.environ.get("FED_REPO_ROOT")
    root = _resolved(env) if env else Path(__file__).resolve().parents[2]
    return _require_dir(root, "Repository root")


def package_root() -> Path:
    """The importable source package, ``<repo>/mnist_dermnist``."""
    return _require_dir(repo_root() / _PACKAGE_NAME, "Package root")


def results_root() -> Path:
    """Experiment results root. Honours ``$FED_RESULTS_ROOT``; else ``<package>/results``."""
    env = os.environ.get("FED_RESULTS_ROOT")
    root = _resolved(env) if env else package_root() / "results"
    return _require_dir(root, "Results root")


def thesis_ready_root() -> Path:
    """The curated ``results/thesis_ready`` bundle."""
    return _require_dir(results_root() / "thesis_ready", "thesis_ready root")


def thesis_data_dir() -> Path:
    """Aggregated thesis tables/data, ``results/thesis_ready/data``."""
    return _require_dir(thesis_ready_root() / "data", "thesis_ready data dir")


def thesis_figures_dir() -> Path:
    """Canonical thesis figure PDFs, ``results/thesis_ready/figures``."""
    return _require_dir(thesis_ready_root() / "figures", "thesis_ready figures dir")
