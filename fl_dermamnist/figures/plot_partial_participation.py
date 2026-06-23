"""Partial-participation convergence figure -- clean 1x2 cross-C comparison.

Reads:
  results/flower_C0_baseline/history_*.csv        (C=1.0 full participation)
  results/system_het_partial_C0.5/history_*.csv   (C=0.5 partial participation)

Plots FedAvg vs FedProx mean +/- SEM validation macro-F1 trajectories
under each of the two participation regimes on the engineered partition.

Output:
  results/thesis_ready/figures/F_partial_participation.{pdf,png}

Style: matches the rest of the chapter -- axes + legend + panel subtitle,
no figure title, no inset notes.
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
    pat = re.compile(rf"history_{algo}_mu{mu}_E20(?:_C[0-9.]+)?_s(\d+)\.csv")
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


def _panel(ax, fa, fp, show_legend=False):
    for curves, col, lbl in [
        (fa, COL_FEDAVG,  "FedAvg"),
        (fp, COL_FEDPROX, "FedProx ($\\mu = 0.01$)"),
    ]:
        rounds, mean, sem = stack(curves)
        ax.plot(rounds, mean, color=col, linewidth=1.6, label=lbl)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=col, alpha=0.20, linewidth=0)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 0.60)
    ax.set_xlabel("Communication round")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    if show_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=9)


def main():
    full_fa  = load_history(RESULTS_ROOT / "flower_C0_baseline",      "fedavg",  "0.0")
    full_fp  = load_history(RESULTS_ROOT / "flower_C0_baseline",      "fedprox", "0.01")
    part_fa  = load_history(RESULTS_ROOT / "system_het_partial_C0.5", "fedavg",  "0.0")
    part_fp  = load_history(RESULTS_ROOT / "system_het_partial_C0.5", "fedprox", "0.01")
    print(f"C=1.0 (full):    n_fa={len(full_fa)}, n_fp={len(full_fp)}")
    print(f"C=0.5 (partial): n_fa={len(part_fa)}, n_fp={len(part_fp)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             sharey=True,
                             gridspec_kw={"wspace": 0.10})
    _panel(axes[0], full_fa, full_fp, show_legend=True)
    axes[0].set_ylabel("Validation macro-F1")
    _panel(axes[1], part_fa, part_fp, show_legend=False)

    axes[0].text(0.5, -0.20, "$C = 1.0$ (full participation)",
                 transform=axes[0].transAxes, ha="center", fontsize=11)
    axes[1].text(0.5, -0.20, "$C = 0.5$ (partial participation)",
                 transform=axes[1].transAxes, ha="center", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_partial_participation.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
