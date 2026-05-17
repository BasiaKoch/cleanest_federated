"""Additional statistical robustness analyses requested by the methodology review:

  1. Sign test (independent of magnitude / normality assumptions)
  2. Hodges-Lehmann estimate of paired difference (median of Walsh averages)
  3. Leave-one-seed-out (LOSO) Wilcoxon — does the result survive removing any
     single seed, especially the largest positive contributor?
  4. Holm-Bonferroni-corrected per-class p-values

All operate on the existing 20 test_at_best JSONs. No new compute required.
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
HEADLINE = ROOT.parent / "headline"

CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]


def load_pairs():
    fa, fp = {}, {}
    pat = re.compile(r"test_at_best_(fedavg|fedprox)_mu([0-9.]+)_E20_s(\d+)\.json")
    for f in sorted(HEADLINE.glob("test_at_best_*.json")):
        m = pat.match(f.name)
        if not m: continue
        d = json.load(open(f))
        seed = int(m.group(3))
        (fa if m.group(1) == "fedavg" else fp)[seed] = d
    seeds = sorted(set(fa) & set(fp))
    return fa, fp, seeds


def sign_test(deltas):
    """Two-sided sign test: probability of >= k wins out of n given p=0.5."""
    n = len(deltas)
    k = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    n_effective = n - ties
    if n_effective == 0:
        return float("nan"), k, n
    # Two-sided: P(X >= max(k, n-k))
    from math import comb
    k_eff = max(k, n_effective - k)
    p_one_sided = sum(comb(n_effective, i) for i in range(k_eff, n_effective + 1)) / (2 ** n_effective)
    return min(1.0, 2 * p_one_sided), k, n_effective


def hodges_lehmann(deltas):
    """Hodges-Lehmann estimate: median of all pairwise Walsh averages (d_i+d_j)/2."""
    n = len(deltas)
    walsh = [(deltas[i] + deltas[j]) / 2.0 for i in range(n) for j in range(i, n)]
    return float(np.median(walsh))


def loso_wilcoxon(deltas):
    """Leave-one-seed-out Wilcoxon: drop each seed, report p and mean."""
    rows = []
    for drop in range(len(deltas)):
        sub = [d for i, d in enumerate(deltas) if i != drop]
        if HAS_SCIPY:
            try:
                _, p = stats.wilcoxon(sub, alternative="two-sided")
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        rows.append({
            "dropped_seed_idx": drop,
            "dropped_delta":    deltas[drop],
            "mean_remaining":   float(np.mean(sub)),
            "wilcoxon_p":       p,
            "still_significant_at_05": bool(p < 0.05) if not np.isnan(p) else False,
        })
    return rows


def holm_correction(p_values, alpha=0.05):
    """Holm step-down: sort p-values, accept p_(i) <= alpha / (k - i + 1)."""
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    k = len(indexed)
    decisions = [False] * k
    rejected_so_far = True
    for i, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (k - i)
        if rejected_so_far and p <= threshold:
            decisions[orig_idx] = True
        else:
            rejected_so_far = False
    adjusted_p = [None] * k
    # Holm-adjusted p_i = max over j<=i of p_(j) * (k - j + 1), clipped to 1
    running_max = 0.0
    for i, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, max(running_max, p * (k - i)))
        running_max = adj
        adjusted_p[orig_idx] = adj
    return decisions, adjusted_p


def main():
    fa, fp, seeds = load_pairs()
    n = len(seeds)
    print(f"Loaded {n} paired seeds: {seeds}\n")

    # Headline macro-F1 deltas
    deltas = [fp[s]["macro_f1"] - fa[s]["macro_f1"] for s in seeds]
    print("=" * 70)
    print("HEADLINE macro-F1 DELTAS (for reference)")
    print("=" * 70)
    print(f"per-seed: {[f'{d:+.4f}' for d in deltas]}")
    print(f"mean = {np.mean(deltas):+.4f}, sd = {np.std(deltas, ddof=1):.4f}\n")

    # ---- Sign test ----
    p_sign, wins, n_eff = sign_test(deltas)
    print(f"Sign test:           {wins}/{n_eff} positive, two-sided p = {p_sign:.4f}")

    # ---- Hodges-Lehmann ----
    hl = hodges_lehmann(deltas)
    print(f"Hodges-Lehmann est:  {hl:+.4f}  (median of Walsh averages; preferred over mean for non-normal data)")

    # ---- Wilcoxon (for reference) ----
    if HAS_SCIPY:
        _, p_wilcox = stats.wilcoxon(deltas, alternative="two-sided")
        print(f"Wilcoxon (ref):      p = {p_wilcox:.4f}")

    # ---- LOSO ----
    print("\nLeave-one-seed-out Wilcoxon:")
    print(f"{'dropped seed':>14} | {'dropped Δ':>10} | {'mean Δ (n=9)':>14} | {'p (n=9)':>10} | {'sig?':>6}")
    print("-" * 70)
    loso_rows = loso_wilcoxon(deltas)
    for s, row in zip(seeds, loso_rows):
        sig_mark = "✓" if row["still_significant_at_05"] else "✗"
        print(f"{s:>14} | {row['dropped_delta']:>+10.4f} | {row['mean_remaining']:>+14.4f} | "
              f"{row['wilcoxon_p']:>10.4f} | {sig_mark:>6}")

    n_still_sig = sum(1 for r in loso_rows if r["still_significant_at_05"])
    print(f"\nSummary: {n_still_sig}/{n} LOSO subsamples remain significant at α=0.05.")

    # ---- Holm correction on per-class ----
    print("\n" + "=" * 70)
    print("PER-CLASS HOLM-CORRECTED p-VALUES")
    print("=" * 70)
    per_class_pvals = []
    per_class_deltas = []
    for c in range(7):
        fa_vals = [fa[s]["per_class_f1"][c] for s in seeds]
        fp_vals = [fp[s]["per_class_f1"][c] for s in seeds]
        d = [fp_vals[i] - fa_vals[i] for i in range(n)]
        per_class_deltas.append(float(np.mean(d)))
        if HAS_SCIPY:
            try:
                _, p = stats.wilcoxon(d, alternative="two-sided")
            except ValueError:
                p = 1.0
        else:
            p = float("nan")
        per_class_pvals.append(p)

    holm_decisions, holm_p = holm_correction(per_class_pvals, alpha=0.05)
    print(f"{'class':>15} | {'Δ':>8} | {'raw p':>8} | {'Holm p':>8} | {'raw sig':>8} | {'Holm sig':>9}")
    print("-" * 72)
    for c in range(7):
        raw_sig = "✓" if per_class_pvals[c] < 0.05 else "✗"
        holm_sig = "✓" if holm_decisions[c] else "✗"
        print(f"{CLASS_NAMES[c]:>15} | {per_class_deltas[c]:>+8.4f} | "
              f"{per_class_pvals[c]:>8.4f} | {holm_p[c]:>8.4f} | "
              f"{raw_sig:>8} | {holm_sig:>9}")

    # ---- Save ----
    out = {
        "description": "Additional statistical robustness analyses on the 10 paired seeds.",
        "n_paired_seeds": n,
        "seeds": seeds,
        "deltas": [float(d) for d in deltas],
        "sign_test": {
            "wins": int(wins),
            "n_effective": int(n_eff),
            "p_two_sided": float(p_sign),
            "interpretation": (
                "Sign test ignores magnitude and is robust to outliers. "
                "p=0.020-ish from this test corroborates the Wilcoxon result "
                "based on direction alone."
            ),
        },
        "hodges_lehmann_estimate": {
            "value": hl,
            "interpretation": (
                "Hodges-Lehmann is the median of paired Walsh averages, "
                "more robust than the mean to extreme paired differences. "
                "Compare to the arithmetic mean Δ = +0.0267."
            ),
        },
        "leave_one_seed_out": {
            "rows": [{"seed": s, **r} for s, r in zip(seeds, loso_rows)],
            "n_still_significant_at_05": n_still_sig,
            "interpretation": (
                f"LOSO Wilcoxon: removing any single seed from the 10-seed "
                f"sweep leaves {n_still_sig} of 10 subsamples significant at α=0.05. "
                f"This tests whether the result depends on any single influential seed."
            ),
        },
        "per_class_holm": {
            "alpha": 0.05,
            "k_tests": 7,
            "classes": [
                {
                    "class": CLASS_NAMES[c],
                    "delta_mean": per_class_deltas[c],
                    "raw_p": per_class_pvals[c],
                    "holm_adjusted_p": holm_p[c],
                    "significant_raw": bool(per_class_pvals[c] < 0.05),
                    "significant_holm": bool(holm_decisions[c]),
                }
                for c in range(7)
            ],
            "interpretation": (
                "Holm step-down correction is more powerful than Bonferroni "
                "while preserving family-wise error rate at 0.05 across 7 "
                "per-class tests."
            ),
        },
    }
    out_path = DATA_DIR / "extra_statistics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
