"""Worst-case per-seed per-class regression analysis.

A reviewer-driven robustness check: even when the *mean* per-class regression
across 10 seeds is small (e.g., vascular Δ = -0.012), an individual seed
could exhibit a catastrophic per-class drop that would be clinically
relevant. This script computes per-(seed, class) Δ F1 = FedProx - FedAvg
and reports:

  - The single worst Δ across all (seed × class) cells
  - The worst Δ per class (across seeds)
  - The worst Δ per seed (across classes)
  - Histogram of all 70 per-(seed, class) Δ values

If the worst per-(seed, class) Δ is more negative than the pre-registered
safety threshold of -0.05, the safety claim needs to be revised.

Reads: mnist_dermnist/results/headline/test_at_best_*.json
Writes: thesis_ready/data/worst_case_per_class.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
HEADLINE  = ROOT.parent / "headline"
CLASSES   = ["actinic", "basal", "benign_kerat", "dermato",
             "melanoma", "mel_nevi", "vascular"]


def load_pairs():
    fa, fp = {}, {}
    pat = re.compile(r"test_at_best_(fedavg|fedprox)_mu[0-9.]+_E20_s(\d+)\.json")
    for f in sorted(HEADLINE.glob("test_at_best_*.json")):
        m = pat.match(f.name)
        if not m: continue
        algo, seed = m.group(1), int(m.group(2))
        d = json.load(open(f))
        (fa if algo == "fedavg" else fp)[seed] = d
    return fa, fp


def main():
    fa, fp = load_pairs()
    seeds = sorted(set(fa) & set(fp))
    n = len(seeds)
    print(f"Loaded {n} paired seeds: {seeds}\n")

    # Build a (seed × class) table of Δ F1 = FedProx - FedAvg
    rows = []
    grid = np.zeros((n, 7), dtype=float)
    for i, s in enumerate(seeds):
        for c in range(7):
            d = fp[s]["per_class_f1"][c] - fa[s]["per_class_f1"][c]
            grid[i, c] = d
            rows.append({"seed": s, "class": CLASSES[c],
                         "fedavg_f1": fa[s]["per_class_f1"][c],
                         "fedprox_f1": fp[s]["per_class_f1"][c],
                         "delta": d})

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "per_seed_per_class_delta.csv", index=False)
    print(f"Wrote {DATA_DIR / 'per_seed_per_class_delta.csv'}  ({len(df)} rows)")

    # --- The single worst (seed, class) cell ---
    worst_idx = np.unravel_index(grid.argmin(), grid.shape)
    worst_seed = seeds[worst_idx[0]]
    worst_class = CLASSES[worst_idx[1]]
    worst_value = float(grid[worst_idx])
    print(f"\n=== Single worst per-(seed, class) Δ across all 70 cells ===")
    print(f"  seed={worst_seed}  class={worst_class}  Δ={worst_value:+.4f}")
    print(f"  FedAvg={fa[worst_seed]['per_class_f1'][worst_idx[1]]:.4f}, "
          f"FedProx={fp[worst_seed]['per_class_f1'][worst_idx[1]]:.4f}")

    # --- Worst Δ per class (across seeds) ---
    print(f"\n=== Worst-seed Δ per class ===")
    print(f"{'class':>14} {'worst Δ':>10} {'at seed':>10} {'mean Δ':>10}")
    print("-" * 50)
    worst_by_class = []
    for c in range(7):
        col = grid[:, c]
        worst_seed_c = seeds[int(col.argmin())]
        worst_v = float(col.min())
        print(f"{CLASSES[c]:>14} {worst_v:>+10.4f} {worst_seed_c:>10} {col.mean():>+10.4f}")
        worst_by_class.append({
            "class": CLASSES[c],
            "worst_delta": worst_v,
            "worst_seed":  worst_seed_c,
            "mean_delta":  float(col.mean()),
            "all_per_seed_deltas": col.tolist(),
        })

    # --- Worst Δ per seed (across classes) ---
    print(f"\n=== Worst-class Δ per seed ===")
    print(f"{'seed':>10} {'worst Δ':>10} {'on class':>16}")
    print("-" * 40)
    worst_by_seed = []
    for i, s in enumerate(seeds):
        row = grid[i, :]
        worst_class_s = CLASSES[int(row.argmin())]
        worst_v = float(row.min())
        print(f"{s:>10} {worst_v:>+10.4f} {worst_class_s:>16}")
        worst_by_seed.append({
            "seed": s, "worst_delta": worst_v,
            "worst_class": worst_class_s,
        })

    # --- Distribution stats ---
    threshold = -0.05
    n_below = int((grid < threshold).sum())
    print(f"\n=== Safety threshold check (pre-registered -0.05) ===")
    print(f"  cells below -0.05:  {n_below} / {grid.size}")
    print(f"  cells below -0.10:  {int((grid < -0.10).sum())} / {grid.size}")
    print(f"  cells below -0.20:  {int((grid < -0.20).sum())} / {grid.size}")
    print(f"  min delta:          {float(grid.min()):+.4f}")
    print(f"  max delta:          {float(grid.max()):+.4f}")

    out = {
        "description": (
            "Worst-case per-(seed, class) test F1 regression analysis. "
            "Computes Δ F1 = FedProx_per_class_F1 - FedAvg_per_class_F1 "
            "for each of 10 seeds × 7 classes = 70 cells. Used to defend "
            "the safety claim that no per-class regression is clinically "
            "catastrophic, not just that the mean is small."
        ),
        "n_seeds": n,
        "n_classes": 7,
        "n_cells": 70,
        "single_worst_cell": {
            "seed": worst_seed, "class": worst_class, "delta": worst_value,
        },
        "worst_by_class": worst_by_class,
        "worst_by_seed":  worst_by_seed,
        "global_stats": {
            "min": float(grid.min()),
            "max": float(grid.max()),
            "mean": float(grid.mean()),
            "median": float(np.median(grid)),
        },
        "safety_threshold_checks": {
            "threshold_at_-0.05_cells_below": n_below,
            "threshold_at_-0.10_cells_below": int((grid < -0.10).sum()),
            "threshold_at_-0.20_cells_below": int((grid < -0.20).sum()),
        },
    }
    out_path = DATA_DIR / "worst_case_per_class.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
