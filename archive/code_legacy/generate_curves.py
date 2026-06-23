"""Multi-seed convergence-curves plot: FedAvg vs FedProx (HPC headline sweep).

Pulls per-round val metrics from all 10 paired-seed history CSVs and produces:
- 4-panel main figure: val_macro_f1, val_loss, train_loss (log), val_balanced_accuracy
- 7-panel per-class figure: val_f1_class_0 ... val_f1_class_6
- Mean line + ±SEM shaded band across seeds for each algorithm

Style: mean-trajectory + shaded band is the convention in FL papers (Li 2020,
Karimireddy 2020 SCAFFOLD, Hsu 2019).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT     = Path(__file__).resolve().parent.parent
FIG_DIR  = ROOT / "figures"
HEADLINE = ROOT.parent / "headline"

CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]
PREVALENCE  = [3.27, 5.13, 10.97, 1.15, 11.11, 67.05, 1.41]

C_FA = "#2b6cb0"
C_FP = "#dd6b20"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def load_all(algo: str) -> pd.DataFrame:
    rows = []
    for f in sorted(HEADLINE.glob(f"history_{algo}_*.csv")):
        df = pd.read_csv(f)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def mean_band(df: pd.DataFrame, col: str):
    agg = df.groupby("round")[col].agg(["mean", "std", "count"])
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    return agg["mean"], sem


def plot_panel(ax, fa: pd.DataFrame, fp: pd.DataFrame, col: str, title: str,
               ylabel: str, log: bool = False, legend: bool = False):
    for df, c, label in [(fa, C_FA, "FedAvg"), (fp, C_FP, "FedProx")]:
        mean, sem = mean_band(df, col)
        rounds = mean.index.values
        ax.plot(rounds, mean.values, color=c, linewidth=2, label=label)
        ax.fill_between(rounds, mean - sem, mean + sem, color=c, alpha=0.20)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)


def savefig(fig, name: str):
    p_png = FIG_DIR / f"{name}.png"
    p_pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_pdf,           bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def fig_08_main_curves(fa, fp):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_panel(axes[0, 0], fa, fp, "val_macro_f1",
               "(A) Validation macro-F1", "Macro-F1", legend=True)
    plot_panel(axes[0, 1], fa, fp, "val_loss",
               "(B) Validation loss", "Cross-entropy")
    plot_panel(axes[1, 0], fa, fp, "train_loss",
               "(C) Training loss (log scale)", "Train loss", log=True)
    plot_panel(axes[1, 1], fa, fp, "val_balanced_accuracy",
               "(D) Validation balanced accuracy", "Balanced accuracy")
    fig.suptitle("FedAvg vs FedProx — convergence curves (mean ± SEM across 10 paired seeds)",
                 fontsize=13, fontweight="bold")
    savefig(fig, "08_curves_main")


def fig_09_per_class_curves(fa, fp):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes_flat = axes.flatten()
    for i, name in enumerate(CLASS_NAMES):
        ax = axes_flat[i]
        col = f"val_f1_class_{i}"
        plot_panel(ax, fa, fp, col,
                   f"({chr(65+i)}) {name}  ({PREVALENCE[i]:.1f}%)",
                   "F1", legend=(i == 0))
        ax.set_ylim(0, 1.0)
    axes_flat[-1].axis("off")  # 7 classes, 8 slots
    fig.suptitle("Per-class validation F1 — FedAvg vs FedProx (mean ± SEM across 10 paired seeds)",
                 fontsize=13, fontweight="bold")
    savefig(fig, "09_curves_per_class")


def fig_10_overfitting_diagnostic(fa, fp):
    """Train + val loss side-by-side per algorithm, highlighting overfitting gap."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for ax, (df, name, c) in zip(axes, [(fa, "FedAvg", C_FA), (fp, "FedProx", C_FP)]):
        tr_mean, tr_sem = mean_band(df, "train_loss")
        vl_mean, vl_sem = mean_band(df, "val_loss")
        rounds = tr_mean.index.values
        ax.plot(rounds, tr_mean, color=c, linewidth=2, label="Train loss")
        ax.fill_between(rounds, tr_mean - tr_sem, tr_mean + tr_sem, color=c, alpha=0.20)
        ax.plot(rounds, vl_mean, color=c, linewidth=2, linestyle="--", label="Val loss")
        ax.fill_between(rounds, vl_mean - vl_sem, vl_mean + vl_sem, color=c, alpha=0.10)
        ax.set_yscale("log")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Loss (log)")
        ax.set_title(f"{name} — overfitting profile")
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Train vs Val loss — overfitting onset", fontsize=13, fontweight="bold")
    savefig(fig, "10_overfitting_diagnostic")


if __name__ == "__main__":
    print("Loading all 20 HPC history CSVs...")
    fa = load_all("fedavg_mu0.0")
    fp = load_all("fedprox_mu0.01")
    print(f"  FedAvg : {fa['seed'].nunique()} seeds × {fa['round'].max()+1} rounds = {len(fa)} rows")
    print(f"  FedProx: {fp['seed'].nunique()} seeds × {fp['round'].max()+1} rounds = {len(fp)} rows")
    print()
    print("Generating curves figures...")
    fig_08_main_curves(fa, fp)
    fig_09_per_class_curves(fa, fp)
    fig_10_overfitting_diagnostic(fa, fp)
    print()
    print("Done.")
