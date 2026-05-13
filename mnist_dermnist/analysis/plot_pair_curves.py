"""Plot FedAvg vs FedProx convergence curves from two history CSVs (single seed).

Usage:
    PYTHONPATH=. python -m mnist_dermnist.analysis.plot_pair_curves \\
        --fedavg  mnist_dermnist/results/cpu_quick/history_fedavg_mu0.0_E20_s42.csv \\
        --fedprox mnist_dermnist/results/cpu_quick/history_fedprox_mu0.01_E20_s42.csv \\
        --out     mnist_dermnist/results/cpu_quick/curves_pair_s42.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"fedavg": "#2b6cb0", "fedprox": "#dd6b20"}     # blue / orange
LABELS = {"fedavg": "FedAvg",   "fedprox": "FedProx"}


def _plot(ax, fa: pd.DataFrame, fp: pd.DataFrame, col: str, title: str,
          ylabel: str | None = None, log: bool = False) -> None:
    for algo, df in [("fedavg", fa), ("fedprox", fp)]:
        ax.plot(df["round"], df[col], linewidth=2,
                color=COLORS[algo], label=LABELS[algo])
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel or title)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)


def _annotate_best(ax, df: pd.DataFrame, col: str, algo: str) -> None:
    idx = df[col].idxmax()
    r, v = int(df.loc[idx, "round"]), float(df.loc[idx, col])
    ax.scatter([r], [v], color=COLORS[algo], s=60, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"best {LABELS[algo]} ({r}, {v:.3f})",
                xy=(r, v), xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=COLORS[algo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedavg", required=True)
    ap.add_argument("--fedprox", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caveat", default=None,
                    help="Optional caveat string shown prominently below suptitle")
    args = ap.parse_args()

    fa = pd.read_csv(args.fedavg)
    fp = pd.read_csv(args.fedprox)

    seed = int(fa["seed"].iloc[0]) if "seed" in fa.columns else None
    mu_fp = float(fp["mu"].iloc[0]) if "mu" in fp.columns else None
    E     = int(fa["local_epochs"].iloc[0]) if "local_epochs" in fa.columns else None

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    _plot(axes[0, 0], fa, fp, "val_macro_f1",
          "(A) Validation macro-F1", ylabel="Macro-F1")
    _annotate_best(axes[0, 0], fa, "val_macro_f1", "fedavg")
    _annotate_best(axes[0, 0], fp, "val_macro_f1", "fedprox")
    axes[0, 0].legend(loc="lower right", fontsize=10)

    _plot(axes[0, 1], fa, fp, "val_loss",
          "(B) Validation loss", ylabel="Cross-entropy")

    _plot(axes[1, 0], fa, fp, "val_balanced_accuracy",
          "(C) Validation balanced accuracy", ylabel="Balanced accuracy")

    _plot(axes[1, 1], fa, fp, "train_loss_weighted",
          "(D) Training loss (size-weighted)", ylabel="Train loss (log)",
          log=True)

    # Headline numbers in suptitle
    fa_best = fa["val_macro_f1"].max()
    fp_best = fp["val_macro_f1"].max()
    delta_best = fp_best - fa_best
    title = "FedAvg vs FedProx — convergence curves (single seed"
    if seed is not None: title += f"={seed}"
    title += f", E={E}, μ_FedProx={mu_fp})\n"
    title += f"best val_macro_f1:  FedAvg {fa_best:.4f}   FedProx {fp_best:.4f}   Δ = {delta_best:+.4f}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    if args.caveat:
        fig.text(0.5, -0.02, args.caveat, ha="center", fontsize=10,
                 color="#c53030", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fed7d7", ec="#c53030"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.pdf')}")
    print()
    print(f"FedAvg  best val macro-F1 = {fa_best:.4f} at round {int(fa.loc[fa['val_macro_f1'].idxmax(), 'round'])}")
    print(f"FedProx best val macro-F1 = {fp_best:.4f} at round {int(fp.loc[fp['val_macro_f1'].idxmax(), 'round'])}")
    print(f"Δ best val macro-F1       = {delta_best:+.4f}")


if __name__ == "__main__":
    main()
