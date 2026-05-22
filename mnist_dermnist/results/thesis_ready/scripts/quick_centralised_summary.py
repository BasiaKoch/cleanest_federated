"""Quick centralised-baseline summary.

Reads every ``centralised_seed*.json`` in
``mnist_dermnist/results/centralised/`` and prints macro-F1
mean / std / range / per-seed. Foundational for the federation-tax
calculation; the full federation-tax analyser (with paired Flower
comparison + CI on the closed-fraction) is the Stage D.3 deliverable
of ``ANALYSIS_PLAN.md``.

Usage:
    PYTHONPATH=. python mnist_dermnist/results/thesis_ready/scripts/quick_centralised_summary.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np


CENTRAL_DIR = Path(__file__).resolve().parents[2] / "centralised"


def main() -> int:
    files = sorted(CENTRAL_DIR.glob("centralised_seed*.json"))
    if not files:
        print(f"No centralised_seed*.json found in {CENTRAL_DIR}")
        return 1

    rows = []
    for p in files:
        d = json.load(open(p))
        seed = d.get("seed", "?")
        macro = d.get("macro_f1", float("nan"))
        epoch = d.get("selected_epoch", "?")
        rows.append((seed, macro, epoch, p.name))

    macros = np.array([r[1] for r in rows], dtype=float)
    print(f"Centralised baseline summary")
    print(f"  n_seeds = {len(rows)}")
    print(f"  macro_f1 mean  = {macros.mean():.4f}")
    print(f"  macro_f1 std   = {macros.std(ddof=1):.4f}" if len(macros) > 1 else "  macro_f1 std = (n=1)")
    print(f"  macro_f1 range = [{macros.min():.4f}, {macros.max():.4f}]")
    print()
    print(f"  per-seed:")
    for seed, macro, epoch, name in sorted(rows, key=lambda r: int(r[0]) if str(r[0]).isdigit() else 0):
        print(f"    seed={seed:>7}  macro_f1={macro:.4f}  selected_epoch={epoch}  ({name})")

    # If we know the flower_C0_baseline FedAvg + FedProx headline numbers,
    # also report the federation tax. This block is best-effort; if the
    # files aren't there yet (because flower_C0 hasn't fully landed) it
    # silently skips.
    try:
        flower_dir = CENTRAL_DIR.parent / "flower_C0_baseline"
        fa = [json.load(open(p))["macro_f1"]
              for p in sorted(flower_dir.glob("test_at_best_fedavg_*.json"))]
        fp = [json.load(open(p))["macro_f1"]
              for p in sorted(flower_dir.glob("test_at_best_fedprox_*.json"))]
        if fa and fp:
            fa_mean = float(np.mean(fa))
            fp_mean = float(np.mean(fp))
            cen_mean = float(macros.mean())
            tax = cen_mean - fa_mean
            closed = (fp_mean - fa_mean) / tax if tax > 0 else float("nan")
            print()
            print(f"  federation tax (centralised - FedAvg)        = {tax:+.4f}")
            print(f"  FedProx closes (FedProx-FedAvg) / tax        = {100*closed:.1f}% of the gap")
            print(f"  (using flower_C0_baseline: n_FedAvg={len(fa)} n_FedProx={len(fp)})")
    except Exception as e:
        print(f"  (federation tax skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
