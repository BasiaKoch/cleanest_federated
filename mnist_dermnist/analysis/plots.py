"""Four required plots from CSV logs:
  A) Validation macro-F1 vs round
  B) Training loss vs round
  C) Per-class test F1 bar plot
  D) Client-class heatmap

Reads:
  - history_<algo>_mu<mu>_E<E>_s<seed>.csv  (per-round metrics)
  - test_at_best_<...>.json                 (test metrics at best-val ckpt)
  - mnist_dermnist/results/partitions/partition_<mode>_seed<seed>_counts.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FA_COLOR = "#3182ce"
FP_COLOR = "#dd6b20"
_HIST_PAT = re.compile(r"history_(?P<algo>fedavg|fedprox)_mu(?P<mu>[0-9.]+)_E(?P<E>\d+)_s(?P<seed>\d+)\.csv")
_TEST_PAT = re.compile(r"test_at_best_(?P<algo>fedavg|fedprox)_mu(?P<mu>[0-9.]+)_E(?P<E>\d+)_s(?P<seed>\d+)\.json")

CLASS_NAMES_SHORT = (
    "actinic", "basal", "benign\nkerat", "dermato", "melanoma", "mel\nnevi", "vascular",
)


def _load_histories(results_dir: Path, E: int) -> dict[str, pd.DataFrame]:
    """Returns {'fedavg': df_all_seeds, 'fedprox': df_all_seeds}."""
    out: dict[str, list[pd.DataFrame]] = {"fedavg": [], "fedprox": []}
    for f in sorted(results_dir.glob("history_*.csv")):
        m = _HIST_PAT.match(f.name)
        if not m or int(m["E"]) != E:
            continue
        df = pd.read_csv(f)
        df["seed"] = int(m["seed"])
        out[m["algo"]].append(df)
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in out.items()}


def _load_test_metrics(results_dir: Path, E: int) -> dict:
    out: dict[str, list[dict]] = {"fedavg": [], "fedprox": []}
    for f in sorted(results_dir.glob("test_at_best_*.json")):
        m = _TEST_PAT.match(f.name)
        if not m or int(m["E"]) != E:
            continue
        with open(f) as fp:
            data = json.load(fp)
            data["seed"] = int(m["seed"])
            out[m["algo"]].append(data)
    return out


def plot_val_macro_f1_vs_round(ax, histories: dict, E: int) -> None:
    for stem, color, label in [("fedavg", FA_COLOR, "FedAvg"),
                                ("fedprox", FP_COLOR, "FedProx (μ=0.1)")]:
        df = histories.get(stem)
        if df is None or df.empty:
            continue
        agg = df.groupby("round")["val_macro_f1"].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        ax.plot(agg["round"], agg["mean"], color=color, linewidth=2, label=f"{label}  (n={int(agg['count'].max())})")
        ax.fill_between(agg["round"], agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                        color=color, alpha=0.20)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global validation macro-F1")
    ax.set_title(f"(A) Validation macro-F1 vs round (E={E}, mean ± SEM across seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)


def plot_train_loss_vs_round(ax, histories: dict, E: int) -> None:
    for stem, color, label in [("fedavg", FA_COLOR, "FedAvg"),
                                ("fedprox", FP_COLOR, "FedProx (μ=0.1)")]:
        df = histories.get(stem)
        if df is None or df.empty or "train_loss" not in df.columns:
            continue
        agg = df.groupby("round")["train_loss"].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        ax.plot(agg["round"], agg["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(agg["round"], agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                        color=color, alpha=0.20)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Training loss (size-weighted across sampled clients)")
    ax.set_title(f"(B) Training loss vs round (E={E})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)


def plot_per_class_test_f1(ax, tests: dict, E: int) -> None:
    fa = tests.get("fedavg", [])
    fp = tests.get("fedprox", [])
    if not fa or not fp:
        ax.text(0.5, 0.5, "no test data", ha="center", va="center", transform=ax.transAxes)
        return
    fa_arr = np.array([r["per_class_f1"] for r in fa])
    fp_arr = np.array([r["per_class_f1"] for r in fp])
    x = np.arange(len(CLASS_NAMES_SHORT))
    w = 0.38
    ax.bar(x - w / 2, fa_arr.mean(0), w, yerr=fa_arr.std(0), capsize=3,
           color=FA_COLOR, edgecolor="white", label=f"FedAvg (n={len(fa)})")
    ax.bar(x + w / 2, fp_arr.mean(0), w, yerr=fp_arr.std(0), capsize=3,
           color=FP_COLOR, edgecolor="white", label=f"FedProx (n={len(fp)})")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=8)
    ax.set_ylabel("Test F1 (mean ± std)")
    ax.set_title(f"(C) Per-class test F1 (E={E})")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")


def plot_client_class_heatmap(ax, partition_counts_path: Path | None) -> None:
    if partition_counts_path is None or not partition_counts_path.exists():
        ax.text(0.5, 0.5, "no partition counts CSV", ha="center", va="center", transform=ax.transAxes)
        return
    df = pd.read_csv(partition_counts_path, index_col=0)
    df = df.drop(columns=["total"], errors="ignore")
    # Short column labels
    df.columns = [c.split("_", 2)[-1][:12] for c in df.columns]
    im = ax.imshow(df.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"C{i}" for i in df.index], fontsize=8)
    # Annotate non-zero entries
    vmax = df.values.max()
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = int(df.values[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                        color="black" if v < vmax * 0.5 else "white")
    ax.set_title("(D) Client-class distribution")
    plt.colorbar(im, ax=ax, shrink=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="mnist_dermnist/results/headline")
    ap.add_argument("--E", type=int, default=20)
    ap.add_argument("--partition-counts",
                    default="mnist_dermnist/results/partitions/partition_medical_skew_7_clients_seed42_counts.csv",
                    help="Path to a partition counts CSV (from data.partition CLI)")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: <results-dir>/analysis/)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results-dir not found: {results_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = _load_histories(results_dir, E=args.E)
    tests = _load_test_metrics(results_dir, E=args.E)
    n_fa = len({d["seed"].iloc[0] for _, d in histories["fedavg"].groupby("seed")}) if not histories["fedavg"].empty else 0
    n_fp = len({d["seed"].iloc[0] for _, d in histories["fedprox"].groupby("seed")}) if not histories["fedprox"].empty else 0
    print(f"Loaded histories: FedAvg n_seeds={n_fa}, FedProx n_seeds={n_fp}")
    if n_fa == 0 or n_fp == 0:
        warnings.warn("No histories for at least one algorithm — plots will be partial.")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plot_val_macro_f1_vs_round(axes[0, 0], histories, args.E)
    plot_train_loss_vs_round(axes[0, 1], histories, args.E)
    plot_per_class_test_f1(axes[1, 0], tests, args.E)
    plot_client_class_heatmap(axes[1, 1], Path(args.partition_counts) if args.partition_counts else None)
    fig.suptitle(f"FedAvg vs FedProx — DermMNIST headline analysis (E={args.E})",
                 fontsize=13, fontweight="bold", y=1.02)
    out_png = out_dir / f"headline_E{args.E}.png"
    out_pdf = out_dir / f"headline_E{args.E}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
