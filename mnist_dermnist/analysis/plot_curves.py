"""Per-round convergence curves: FedAvg vs FedProx on TWO partitions.

Requires `history_*.csv` files (saved by the updated `run_cpu_flower.py`).
Older runs that didn't save history won't show up here; use
`plot_partition_compare.py` for those.

Outputs a 2×2 figure:
  (A) Validation macro-F1 vs round    — primary metric
  (B) Validation loss vs round         — convergence diagnostic
  (C) Training loss vs round (weighted across clients)
  (D) Validation balanced accuracy vs round
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Colour scheme keeps FedAvg/FedProx pair, varies by partition (light vs dark)
COLORS = {
    ("old", "fedavg"):  "#90cdf4",   # light blue
    ("old", "fedprox"): "#fbd38d",   # light orange
    ("new", "fedavg"):  "#2b6cb0",   # dark blue
    ("new", "fedprox"): "#dd6b20",   # dark orange
}
LINESTYLES = {"fedavg": "--", "fedprox": "-"}

_PAT = re.compile(r"history_(?P<algo>fedavg|fedprox)_mu(?P<mu>[0-9.]+)_E(?P<E>\d+)_s(?P<seed>\d+)\.csv")


def _load_histories(d: Path) -> dict[str, pd.DataFrame]:
    """Return {algorithm: dataframe-concatenated-over-seeds}."""
    out: dict[str, list[pd.DataFrame]] = {"fedavg": [], "fedprox": []}
    for f in sorted(d.glob("history_*.csv")):
        m = _PAT.match(f.name)
        if not m:
            continue
        df = pd.read_csv(f)
        df["seed"] = int(m["seed"])
        df["algorithm"] = m["algo"]
        out[m["algo"]].append(df)
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in out.items()}


def _plot_metric(ax, histories_old: dict, histories_new: dict, col: str, title: str):
    for tag, hist_dict in [("old", histories_old), ("new", histories_new)]:
        for algo, df in hist_dict.items():
            if df is None or df.empty or col not in df.columns:
                continue
            agg = df.groupby("round")[col].agg(["mean", "std", "count"]).reset_index()
            if agg["count"].max() > 1:
                # multi-seed: shade ±SEM
                sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
                ax.fill_between(agg["round"], agg["mean"] - sem, agg["mean"] + sem,
                                color=COLORS[(tag, algo)], alpha=0.18)
            ax.plot(agg["round"], agg["mean"],
                    linewidth=2, linestyle=LINESTYLES[algo],
                    color=COLORS[(tag, algo)],
                    label=f"{algo} ({'balanced_specialist' if tag == 'old' else 'balanced_paired'})")
    ax.set_xlabel("Round")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", default="mnist_dermnist/results/cpu_flower",
                    help="Result dir for balanced_specialist runs")
    ap.add_argument("--new-dir", default="mnist_dermnist/results/cpu_flower_paired",
                    help="Result dir for balanced_paired runs")
    ap.add_argument("--out", default="mnist_dermnist/results/curves_partitions.png")
    args = ap.parse_args()

    old = _load_histories(Path(args.old_dir))
    new = _load_histories(Path(args.new_dir))

    n_old = sum(len(v) for v in old.values())
    n_new = sum(len(v) for v in new.values())
    print(f"Loaded {n_old} rows from {args.old_dir}")
    print(f"Loaded {n_new} rows from {args.new_dir}")
    if n_old + n_new == 0:
        raise SystemExit(
            "No history_*.csv files found. Re-run run_cpu_flower.py with the updated "
            "script (which now saves history) before plotting curves."
        )

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _plot_metric(axes[0, 0], old, new, "val_macro_f1",         "Validation macro-F1")
    _plot_metric(axes[0, 1], old, new, "val_loss",             "Validation loss")
    _plot_metric(axes[1, 0], old, new, "train_loss_weighted",  "Training loss (size-weighted)")
    _plot_metric(axes[1, 1], old, new, "val_balanced_accuracy", "Validation balanced accuracy")
    # Single legend on the first subplot
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Convergence curves — old (balanced_specialist) vs new (balanced_paired) partition",
                 fontsize=12, fontweight="bold", y=1.02)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
