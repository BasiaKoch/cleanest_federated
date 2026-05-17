"""Per-client specialty analysis from existing 20 test_at_best JSONs.

The `balanced_paired_7_clients` partition creates 3 specialty PAIRS plus a
nevi-only generalist:
    Pair 0 = {C0, C1}: actinic + basal       (specialty classes 0, 1)
    Pair 1 = {C2, C3}: benign_kerat + dermato (specialty classes 2, 3)
    Pair 2 = {C4, C5}: melanoma + vascular    (specialty classes 4, 6)
    Generalist = {C6}: mel_nevi only          (class 5)

We compute, for each seed and each algorithm:
    pair_specialty_f1 = mean of test per-class F1 over the pair's minority
                        classes (excluding the universal mel_nevi)

Then aggregate across 10 paired seeds with paired Wilcoxon p-values.

Marija's thesis evaluates individual-client local models against a federated
global model. Our setup uses a single global model evaluated once on the global
test set; the per-client question therefore becomes: how well does the global
model perform on the classes each specialty pair was responsible for?

Reads:   mnist_dermnist/results/headline/test_at_best_*.json
Writes:  thesis_ready/data/per_client_specialty.csv
         thesis_ready/data/per_client_specialty.json
"""
from __future__ import annotations

import json
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HEADLINE = ROOT.parent / "headline"

# Class indices in the test_per_class_f1 array
CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]

# Specialty pairs from balanced_paired_7_clients (excluding mel_nevi=majority)
PAIRS = {
    "pair0 (C0+C1)":         {"classes": [0, 1],      "labels": ["actinic", "basal"]},
    "pair1 (C2+C3)":         {"classes": [2, 3],      "labels": ["benign_kerat", "dermato"]},
    "pair2 (C4+C5)":         {"classes": [4, 6],      "labels": ["melanoma", "vascular"]},
    "generalist (C6)":       {"classes": [5],         "labels": ["mel_nevi"]},
    "all minorities":        {"classes": [0,1,2,3,4,6], "labels": ["all except mel_nevi"]},
}


def main():
    fa_runs, fp_runs = [], []
    pat = re.compile(r"test_at_best_(fedavg|fedprox)_mu([0-9.]+)_E20_s(\d+)\.json")

    for f in sorted(HEADLINE.glob("test_at_best_*.json")):
        m = pat.match(f.name)
        if not m: continue
        d = json.load(open(f))
        d["__seed"] = int(m.group(3))
        d["__algo"] = m.group(1)
        (fa_runs if m.group(1) == "fedavg" else fp_runs).append(d)

    print(f"Loaded {len(fa_runs)} FedAvg, {len(fp_runs)} FedProx runs")
    fa = {r["__seed"]: r for r in fa_runs}
    fp = {r["__seed"]: r for r in fp_runs}
    seeds = sorted(set(fa) & set(fp))
    n = len(seeds)
    print(f"Paired seeds: {n}\n")

    def pair_f1(run, classes):
        """Mean of per-class F1 over the specified class indices."""
        pcf1 = run["per_class_f1"]
        return float(np.mean([pcf1[c] for c in classes]))

    rows = []
    print(f"{'Specialty':>22} | {'classes':>22} | {'FedAvg':>8} | {'FedProx':>8} | "
          f"{'Δ':>8} | {'wins':>6} | {'wilcox p':>10}")
    print("-" * 110)
    summary = {}
    for pair_name, pair_info in PAIRS.items():
        fa_vals = [pair_f1(fa[s], pair_info["classes"]) for s in seeds]
        fp_vals = [pair_f1(fp[s], pair_info["classes"]) for s in seeds]
        deltas  = [fp_vals[i] - fa_vals[i] for i in range(n)]
        wins    = sum(1 for d in deltas if d > 0)

        if HAS_SCIPY and any(d != 0 for d in deltas):
            try:
                _, p = stats.wilcoxon(deltas, alternative="two-sided")
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")

        label = ", ".join(pair_info["labels"]) if len(pair_info["labels"]) < 4 \
                else f"{len(pair_info['labels'])} classes"
        print(f"{pair_name:>22} | {label:>22} | {np.mean(fa_vals):>8.4f} | "
              f"{np.mean(fp_vals):>8.4f} | {np.mean(deltas):>+8.4f} | "
              f"{wins:>3d}/{n} | {p:>10.4f}")

        rows.append({
            "specialty":         pair_name,
            "classes":           label,
            "n_classes":         len(pair_info["classes"]),
            "fedavg_mean_f1":    np.mean(fa_vals),
            "fedavg_sd_f1":      np.std(fa_vals, ddof=1),
            "fedprox_mean_f1":   np.mean(fp_vals),
            "fedprox_sd_f1":     np.std(fp_vals, ddof=1),
            "delta_mean":        np.mean(deltas),
            "delta_sd":          np.std(deltas, ddof=1),
            "fedprox_wins":      wins,
            "n_pairs":           n,
            "wilcoxon_p":        p,
            "significant_05":    bool(p < 0.05) if not np.isnan(p) else False,
        })
        summary[pair_name] = rows[-1].copy()

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "per_client_specialty.csv", index=False)
    with open(DATA_DIR / "per_client_specialty.json", "w") as f:
        json.dump({
            "description": "Per-specialty-pair test F1 from the global model. "
                           "Each row aggregates over the classes a client pair "
                           "(or generalist) was responsible for in the "
                           "balanced_paired_7_clients partition.",
            "n_paired_seeds": n,
            "seeds": seeds,
            "specialties": summary,
        }, f, indent=2)

    print(f"\nWrote {DATA_DIR / 'per_client_specialty.csv'}")
    print(f"Wrote {DATA_DIR / 'per_client_specialty.json'}")


if __name__ == "__main__":
    main()
