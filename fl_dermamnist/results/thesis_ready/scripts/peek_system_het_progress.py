"""Descriptive peek at the system-heterogeneity sweep progress.

Reports per-(condition, algorithm) macro-F1 mean ± std at whatever
n is currently on disk. NOT an inferential analysis — no paired
Wilcoxon, no H2 contrast, no significance claim. Useful when the
sweeps are still in flight and you want to see "are they finishing
at plausible numbers?" without biasing the eventual canonical
analysis (which is analyse_system_het.py once everything has
landed).

Usage:
    PYTHONPATH=. python fl_dermamnist/results/thesis_ready/scripts/peek_system_het_progress.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parents[2]
SWEEPS = [
    ("flower_C0_baseline",         30, ["fedavg", "fedprox", "fednova"]),
    ("system_het_fixed",           20, ["fedavg", "fedprox"]),
    ("system_het_random",          20, ["fedavg", "fedprox"]),
    ("system_het_random_fednova",  10, ["fednova"]),
]


def _read_macros(directory: Path, algo: str) -> list[float]:
    """Return list of macro_f1 values for the given algo in directory."""
    out = []
    for p in sorted(directory.glob("test_at_best_*.json")):
        if f"_{algo}_" not in p.name:
            continue
        try:
            out.append(float(json.load(open(p))["macro_f1"]))
        except Exception:
            pass
    return out


def main() -> int:
    print("=" * 82)
    print(f"  Descriptive system-het progress (peek only — NOT the canonical analysis)")
    print("=" * 82)
    print(f"  {'sweep':32s}  {'algo':8s}  {'n':>4s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}")
    print("  " + "-" * 78)
    for dirname, expected, algos in SWEEPS:
        d = RESULTS / dirname
        if not d.is_dir():
            print(f"  {dirname:32s}  (directory missing)")
            continue
        for algo in algos:
            macros = _read_macros(d, algo)
            n = len(macros)
            if n == 0:
                print(f"  {dirname:32s}  {algo:8s}  {n:>4d}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>8s}")
                continue
            mean = float(np.mean(macros))
            std  = float(np.std(macros, ddof=1)) if n > 1 else 0.0
            print(f"  {dirname:32s}  {algo:8s}  {n:>4d}  "
                  f"{mean:>8.4f}  {std:>8.4f}  {min(macros):>8.4f}  {max(macros):>8.4f}")
    print()
    print("  Note: these are UNPAIRED descriptive means at the current state of the sweep.")
    print("  The canonical inferential analysis (paired Wilcoxon on Δ, H2 contrast, per-class")
    print("  breakdown) is in thesis_ready_system_het/scripts/analyse_system_het.py and")
    print("  should be run once all four directories reach their expected counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
