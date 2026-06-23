"""Partition robustness convergence figure -- clean 2x2 grid.

Four partition designs side-by-side on the Flower runtime:
  IID
  Dirichlet alpha = 0.1
  Specialist (1-of-7)
  Engineered balanced-paired

Each panel: FedAvg vs FedProx mean +/- SEM trajectories across the
ten paired seeds. Same minimal aesthetic as the IID and engineered
convergence figures.

Output:
  results/thesis_ready/figures/F_partition_robustness.{pdf,png}
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
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_history(sweep_dir: Path, algo: str, mu: str) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"history_{algo}_mu{mu}_E20_s(\d+)\.csv")
    for f in sorted(sweep_dir.glob(f"history_{algo}_*.csv")):
        m = pat.match(f.name)
        if not m: continue
        out[int(m.group(1))] = pd.read_csv(f)[["round", "val_macro_f1"]].copy()
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


def _panel(ax, fa_curves, fp_curves, title, show_legend=False):
    for algo, curves, col, lbl in [
        ("fedavg",  fa_curves, COL_FEDAVG,  "FedAvg"),
        ("fedprox", fp_curves, COL_FEDPROX, "FedProx ($\\mu = 0.01$)"),
    ]:
        rounds, mean, sem = stack(curves)
        ax.plot(rounds, mean, color=col, linewidth=1.5, label=lbl)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=col, alpha=0.20, linewidth=0)
    ax.set_title(title, loc="left", pad=4, fontsize=11, fontweight="bold")
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 0.65)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    if show_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=9)


def main():
    sweeps = [
        ("IID",                          RESULTS_ROOT / "flower_C0_iid_baseline"),
        ("Dirichlet $\\alpha = 0.1$",    RESULTS_ROOT / "dirichlet_a01"),
        ("Specialist (1-of-7)",          RESULTS_ROOT / "specialist_partition"),
        ("Engineered balanced-paired",   RESULTS_ROOT / "flower_C0_baseline"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.30, "wspace": 0.10})
    axes_flat = axes.flatten()

    for i, (label, sweep_dir) in enumerate(sweeps):
        fa = load_history(sweep_dir, "fedavg",  "0.0")
        fp = load_history(sweep_dir, "fedprox", "0.01")
        print(f"{label}: n_fa={len(fa)}, n_fp={len(fp)}")
        _panel(axes_flat[i], fa, fp, label, show_legend=(i == 0))

    # Axis labels
    for ax in axes[1]:
        ax.set_xlabel("Communication round")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation macro-F1")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_partition_robustness.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
