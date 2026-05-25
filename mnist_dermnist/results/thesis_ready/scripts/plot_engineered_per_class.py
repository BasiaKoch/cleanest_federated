"""Engineered partition per-class convergence figure -- clean 2x4 grid.

Reads results/headline/history_*.csv (pure-PyTorch primary, 10 paired seeds)
and plots per-class validation F1 trajectories for the 7 DermaMNIST classes
in a 2x4 grid (7 class panels + 1 legend panel).

Each panel:
  - Class name + prevalence as title
  - Test-set Δ (FedProx - FedAvg) as a small subtitle inside the panel
  - FedAvg vs FedProx mean ± SEM bands across 10 seeds
  - Shared y-axis [0, 1] for visual comparability across classes

Output:
  results/thesis_ready/figures/F_engineered_per_class.{pdf,png}

Style: clean (no figure title, no inset notes), matches the style of
F_iid_convergence and F_engineered_convergence. LaTeX caption carries the
description.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS_ROOT = REPO_ROOT / "mnist_dermnist" / "results"
SWEEP = RESULTS_ROOT / "headline"            # pure-PyTorch primary
OUT_FIG = RESULTS_ROOT / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150

COL_FEDAVG  = "#7FBF94"
COL_FEDPROX = "#3D5A80"

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]
CLASS_PREV = [3.27, 5.13, 10.97, 1.15, 11.11, 67.05, 1.41]


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_history(algo: str, mu: str) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"history_{algo}_mu{mu}_E20_s(\d+)\.csv")
    for f in sorted(SWEEP.glob(f"history_{algo}_*.csv")):
        m = pat.match(f.name)
        if not m: continue
        out[int(m.group(1))] = pd.read_csv(f)
    return out


def load_test_delta() -> list[float]:
    """Per-class mean test-set Delta from test_at_best JSONs."""
    fa, fp = {}, {}
    for f in SWEEP.glob("test_at_best_fedavg_*.json"):
        d = json.load(open(f)); fa[d["seed"]] = d["per_class_f1"]
    for f in SWEEP.glob("test_at_best_fedprox_*.json"):
        d = json.load(open(f)); fp[d["seed"]] = d["per_class_f1"]
    seeds = sorted(set(fa) & set(fp))
    deltas = []
    for c in range(7):
        d_c = [fp[s][c] - fa[s][c] for s in seeds]
        deltas.append(float(np.mean(d_c)))
    return deltas


def stack(curves: dict[int, pd.DataFrame], ycol: str):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    mat = np.full((len(curves), NUM_ROUNDS), np.nan)
    for i, df in enumerate(curves.values()):
        for _, row in df.iterrows():
            r = int(row["round"])
            if 1 <= r <= NUM_ROUNDS:
                mat[i, r - 1] = float(row[ycol])
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return rounds, mean, sem


def main():
    fa = load_history("fedavg",  "0.0")
    fp = load_history("fedprox", "0.01")
    deltas = load_test_delta()
    print(f"FedAvg seeds:  {len(fa)}")
    print(f"FedProx seeds: {len(fp)}")
    for c in range(7):
        print(f"  Class {c} ({CLASS_NAMES[c]:<22}): Δ_test = {deltas[c]:+.4f}")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.34, "wspace": 0.12})
    axes_flat = axes.flatten()

    for c in range(7):
        ax = axes_flat[c]
        ycol = f"val_f1_class_{c}"
        for algo, curves, col, lbl in [
            ("fedavg",  fa, COL_FEDAVG,  "FedAvg"),
            ("fedprox", fp, COL_FEDPROX, "FedProx ($\\mu=0.01$)"),
        ]:
            rounds, mean, sem = stack(curves, ycol)
            ax.plot(rounds, mean, color=col, linewidth=1.4, label=lbl)
            ax.fill_between(rounds, mean - sem, mean + sem,
                            color=col, alpha=0.20, linewidth=0)

        ax.set_title(f"{CLASS_NAMES[c]} ({CLASS_PREV[c]:.2f}%)",
                     loc="left", pad=4, fontsize=10, fontweight="bold")
        ax.text(0.97, 0.06, f"$\\Delta = {deltas[c]:+.3f}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="lightgrey",
                          boxstyle="round,pad=0.25"))
        ax.set_xlim(1, NUM_ROUNDS)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    # Last cell -> legend only
    ax = axes_flat[7]
    ax.axis("off")
    # Draw two sample lines for the legend
    handles = [
        plt.Line2D([], [], color=COL_FEDAVG,  linewidth=2.5, label="FedAvg"),
        plt.Line2D([], [], color=COL_FEDPROX, linewidth=2.5, label="FedProx ($\\mu = 0.01$)"),
    ]
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=12)

    # Common axis labels (one per row/column for compactness)
    for ax in axes[1]:
        ax.set_xlabel("Communication round")
    axes[0, 0].set_ylabel("Validation F1")
    axes[1, 0].set_ylabel("Validation F1")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_engineered_per_class.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
