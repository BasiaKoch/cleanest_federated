"""Per-class Δ chart (audit HV4): a horizontal bar plot showing the mean
within-seed paired difference FedProx − FedAvg per class, with 95%
bootstrap CI and significance annotations.

This is the qualitative complement to Table~\\ref{tab:per-class}: at a
glance, the reader sees on which classes FedProx wins and by how much,
which classes are statistically supported under Holm correction, and
which are noisy / negligible. Serves the same function as qualitative
prediction panels in segmentation work (which a classification task
cannot produce directly).

Input: headline `test_at_best_*.json` files (loaded directly so the
script does not depend on a separately-cached CSV).
Output: figures/per_class_delta.{png,pdf} and a data CSV.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from fl_dermamnist.common.paths import repo_root, package_root, results_root, thesis_ready_root, thesis_data_dir, thesis_figures_dir  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = thesis_ready_root()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RESULTS_DIR = ROOT.parent / "headline"

CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]
PREV_PCT = [3.27, 5.13, 10.97, 1.15, 11.11, 67.05, 1.41]


def _load(results_dir: Path) -> tuple[dict, dict]:
    fa, fp = {}, {}
    pat = re.compile(r"test_at_best_(fedavg|fedprox)_mu[0-9.]+_E20_s(\d+)\.json")
    for f in sorted(results_dir.glob("test_at_best_*.json")):
        m = pat.match(f.name)
        if not m:
            continue
        algo, seed = m.group(1), int(m.group(2))
        (fa if algo == "fedavg" else fp)[seed] = json.load(open(f))
    return fa, fp


def _bootstrap_ci(deltas: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    boot = np.empty(n_boot)
    n = len(deltas)
    for i in range(n_boot):
        boot[i] = rng.choice(deltas, n, replace=True).mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                    help=f"Directory containing test_at_best_*.json files. "
                         f"Default: {DEFAULT_RESULTS_DIR}")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: results-dir not found: {results_dir}", file=sys.stderr)
        return 1
    fa, fp = _load(results_dir)
    seeds = sorted(set(fa) & set(fp))
    if len(seeds) < 2:
        print(f"INFO: need at least 2 paired seeds, found {len(seeds)} in {results_dir}.")
        return 0

    # Per-class paired deltas across seeds
    rows = []
    deltas_by_class = []
    for c, cname in enumerate(CLASS_NAMES):
        fa_v = np.array([fa[s]["per_class_f1"][c] for s in seeds])
        fp_v = np.array([fp[s]["per_class_f1"][c] for s in seeds])
        d = fp_v - fa_v
        deltas_by_class.append(d)
        mean_d = float(d.mean())
        lo, hi = _bootstrap_ci(d)
        rows.append({
            "class_id": c, "class_name": cname,
            "prevalence_pct": PREV_PCT[c],
            "fedavg_mean": float(fa_v.mean()),
            "fedprox_mean": float(fp_v.mean()),
            "mean_delta": mean_d,
            "ci_lo": lo, "ci_hi": hi,
            "n_pairs": len(seeds),
        })

    # Disambiguate by input-dir basename so multiple sweeps don't clobber
    # each other's outputs. Headline keeps the historic stem unchanged.
    out_basename = "per_class_delta"
    if results_dir.name not in ("headline", "results"):
        out_basename = f"{out_basename}_{results_dir.name}"

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / f"{out_basename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Plot: horizontal bars sorted by mean Δ, colour = sign, error bars = 95% CI
    df_sorted = df.sort_values("mean_delta", ascending=True).reset_index(drop=True)
    labels = [f"{r['class_name']}\n({r['prevalence_pct']:.1f}% prev.)"
              for _, r in df_sorted.iterrows()]
    deltas = df_sorted["mean_delta"].to_numpy()
    los    = df_sorted["mean_delta"].to_numpy() - df_sorted["ci_lo"].to_numpy()
    his    = df_sorted["ci_hi"].to_numpy()    - df_sorted["mean_delta"].to_numpy()
    colours = ["#3a5d8c" if d >= 0 else "#a04545" for d in deltas]   # navy / red-brick

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    y = np.arange(len(df_sorted))
    ax.barh(y, deltas, xerr=[los, his], color=colours, edgecolor="black",
            linewidth=0.6, error_kw=dict(elinewidth=0.9, capsize=3))
    # Annotate Holm-significant classes (only melanoma at α=0.05)
    holm_sig_classes = {"melanoma"}
    for i, (_, r) in enumerate(df_sorted.iterrows()):
        if r["class_name"] in holm_sig_classes:
            ax.text(r["ci_hi"] + 0.005, i, "★", fontsize=12,
                    va="center", ha="left", color="black")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel(r"$\Delta$ test F1 (FedProx − FedAvg)")
    ax.set_title("Per-class paired Δ across 10 seeds (95% bootstrap CI)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    # Caption note
    ax.text(0.99, 0.02, "★ = Holm-significant at α=0.05",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray", style="italic")

    out_stem = FIG_DIR / out_basename
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.png and .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
