"""IID convergence figure -- clean single-panel FedAvg vs FedProx.

Reads results/flower_C0_iid_baseline/history_*.csv (10 paired seeds,
uniform local epochs, no system heterogeneity) and plots mean
validation macro-F1 over rounds for each algorithm with +/-SEM
shaded bands.

Output:
  results/thesis_ready/figures/F_iid_convergence.{pdf,png}

Style choices: no plot title (the LaTeX caption carries the
description), no annotations, no inset text. Axis labels + legend
only -- intentionally spartan for the thesis figure aesthetic.
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
SWEEP = RESULTS_ROOT / "flower_C0_iid_baseline"
OUT_FIG = RESULTS_ROOT / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150

COL_FEDAVG  = "#7FBF94"   # soft mint green
COL_FEDPROX = "#3D5A80"   # dark slate blue


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_history(algo: str, mu: str) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"history_{algo}_mu{mu}_E20_s(\d+)\.csv")
    for f in sorted(SWEEP.glob(f"history_{algo}_*.csv")):
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


def main():
    fa = load_history("fedavg",  "0.0")
    fp = load_history("fedprox", "0.01")
    print(f"FedAvg seeds:  {len(fa)}")
    print(f"FedProx seeds: {len(fp)}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for algo, curves, col, lbl in [
        ("fedavg",  fa, COL_FEDAVG,  "FedAvg"),
        ("fedprox", fp, COL_FEDPROX, "FedProx ($\\mu = 0.01$)"),
    ]:
        rounds, mean, sem = stack(curves)
        ax.plot(rounds, mean, color=col, linewidth=1.6, label=lbl)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=col, alpha=0.20, linewidth=0)

    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_iid_convergence.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
