"""System-heterogeneity engineered-partition convergence figure -- clean 1x3.

Reads:
  results/flower_C0_baseline/history_*.csv  (C0 baseline, uniform E=20)
  results/system_het_fixed/history_*.csv    (C1 fixed stragglers)
  results/system_het_random/history_*.csv   (C2 random stragglers)

Plots FedAvg vs FedProx mean +/- SEM validation macro-F1 trajectories
under each of the three system-heterogeneity conditions on the
engineered partition.

Output:
  results/thesis_ready/figures/F_sh_engineered_convergence.{pdf,png}

Style: matches the rest of the chapter -- axes + legend + panel
subtitle, no figure title, no inset notes.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
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
    pat = re.compile(rf"history_{algo}_mu{mu}_E20(?:_sh-[a-z_]+)?_s(\d+)\.csv")
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
    sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return rounds, mean, sem


def _panel(ax, fa, fp, title, show_legend=False):
    for algo, curves, col, lbl in [
        ("fedavg",  fa, COL_FEDAVG,  "FedAvg"),
        ("fedprox", fp, COL_FEDPROX, "FedProx ($\\mu = 0.01$)"),
    ]:
        rounds, mean, sem = stack(curves)
        ax.plot(rounds, mean, color=col, linewidth=1.6, label=lbl)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=col, alpha=0.20, linewidth=0)
    ax.set_title(title, loc="left", pad=4, fontsize=11, fontweight="bold")
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Communication round")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    if show_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=9)


def main():
    sweeps = [
        ("C0 -- no system heterogeneity",    RESULTS_ROOT / "flower_C0_baseline"),
        ("C1 -- fixed stragglers",           RESULTS_ROOT / "system_het_fixed"),
        ("C2 -- random stragglers",          RESULTS_ROOT / "system_het_random"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             sharey=True,
                             gridspec_kw={"wspace": 0.10})

    for i, (label, d) in enumerate(sweeps):
        fa = load_history(d, "fedavg",  "0.0")
        fp = load_history(d, "fedprox", "0.01")
        print(f"{label}: n_fa={len(fa)}, n_fp={len(fp)}")
        _panel(axes[i], fa, fp, label, show_legend=(i == 0))

    axes[0].set_ylabel("Validation macro-F1")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_sh_engineered_convergence.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
