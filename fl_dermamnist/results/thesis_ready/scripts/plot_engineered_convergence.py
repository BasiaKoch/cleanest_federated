"""Engineered-partition convergence figure -- clean 2-panel pure-PyTorch vs Flower.

Reads:
  results/headline/history_*.csv            (pure-PyTorch primary)
  results/flower_C0_baseline/history_*.csv  (Flower replication)

Output:
  results/thesis_ready/figures/F_engineered_convergence.{pdf,png}

Style: same minimal aesthetic as F_iid_convergence -- axes labels and
legend only, no inset text, no titles. The LaTeX caption carries the
description.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS_ROOT = REPO_ROOT / "fl_dermamnist" / "results"
OUT_FIG = RESULTS_ROOT / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150

COL_FEDAVG  = "#7FBF94"
COL_FEDPROX = "#3D5A80"


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_history(sweep_dir: Path, algo: str, mu: str) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"history_{algo}_mu{mu}_E20_s(\d+)\.csv")
    for f in sorted(sweep_dir.glob(f"history_{algo}_*.csv")):
        m = pat.match(f.name)
        if not m:
            continue
        df = pd.read_csv(f)
        out[int(m.group(1))] = df[["round", "val_macro_f1"]].copy()
    return out


def stack(curves: dict[int, pd.DataFrame]):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    mat = np.full((len(curves), NUM_ROUNDS), np.nan)
    for i, df in enumerate(curves.values()):
        for _, row in df.iterrows():
            r = int(row["round"])
            if 1 <= r <= NUM_ROUNDS:
                mat[i, r - 1] = float(row["val_macro_f1"])
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return rounds, mean, sem


def _panel(ax, fa_curves, fp_curves):
    for algo, curves, col, lbl in [
        ("fedavg",  fa_curves, COL_FEDAVG,  "FedAvg"),
        ("fedprox", fp_curves, COL_FEDPROX, "FedProx ($\\mu = 0.01$)"),
    ]:
        rounds, mean, sem = stack(curves)
        ax.plot(rounds, mean, color=col, linewidth=1.6, label=lbl)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=col, alpha=0.20, linewidth=0)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Communication round")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)


def main():
    pt_fa  = load_history(RESULTS_ROOT / "headline",            "fedavg",  "0.0")
    pt_fp  = load_history(RESULTS_ROOT / "headline",            "fedprox", "0.01")
    fl_fa  = load_history(RESULTS_ROOT / "flower_C0_baseline",  "fedavg",  "0.0")
    fl_fp  = load_history(RESULTS_ROOT / "flower_C0_baseline",  "fedprox", "0.01")
    print(f"Pure-PyTorch: n_fa={len(pt_fa)}, n_fp={len(pt_fp)}")
    print(f"Flower:       n_fa={len(fl_fa)}, n_fp={len(fl_fp)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.18},
                             sharey=True)
    _panel(axes[0], pt_fa, pt_fp)
    axes[0].set_ylabel("Validation macro-F1")
    _panel(axes[1], fl_fa, fl_fp)

    # Minimal subtitle text below each panel -- one short label only
    axes[0].text(0.5, -0.20, "Pure-PyTorch (reference)",
                 transform=axes[0].transAxes, ha="center", fontsize=11)
    axes[1].text(0.5, -0.20, "Flower (simulation)",
                 transform=axes[1].transAxes, ha="center", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_engineered_convergence.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
