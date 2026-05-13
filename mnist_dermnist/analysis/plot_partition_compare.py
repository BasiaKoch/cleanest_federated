"""Compare two partition results side by side.

Reads two `flower_comparison_*.json` files and produces a single multi-panel
figure showing:
  (A) Macro-F1 + balanced accuracy + accuracy bars (4 algos × 3 metrics)
  (B) Per-class F1 grouped bars (4 algos × 7 classes)
  (C) Per-class FedProx − FedAvg Δ (2 partitions overlaid)

This is the "final-results" comparison. For per-round CONVERGENCE curves you
need `history_*.csv` from a re-run — the older runs didn't save those.

Usage
-----
    PYTHONPATH=. python -m mnist_dermnist.analysis.plot_partition_compare
    PYTHONPATH=. python -m mnist_dermnist.analysis.plot_partition_compare \\
        --old mnist_dermnist/results/cpu_flower/flower_comparison_seed42_mu0.01.json \\
        --new mnist_dermnist/results/cpu_flower_paired/flower_comparison_seed42_mu0.01.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ("actinic", "basal", "benign\nkerat", "dermato",
               "melanoma", "mel\nnevi", "vascular")

# Colors keep the FedAvg/FedProx pair, varied by partition (light vs dark)
COLORS = {
    ("old", "fedavg"):  "#90cdf4",   # light blue
    ("old", "fedprox"): "#fbd38d",   # light orange
    ("new", "fedavg"):  "#2b6cb0",   # dark blue
    ("new", "fedprox"): "#dd6b20",   # dark orange
}

LABELS = {
    ("old", "fedavg"):  "FedAvg  (balanced_specialist)",
    ("old", "fedprox"): "FedProx (balanced_specialist)",
    ("new", "fedavg"):  "FedAvg  (balanced_paired)",
    ("new", "fedprox"): "FedProx (balanced_paired)",
}


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def panel_summary_bars(ax, old: dict, new: dict) -> None:
    metrics = [
        ("test_macro_f1",         "Macro-F1"),
        ("test_balanced_accuracy", "Balanced acc."),
        ("test_accuracy",          "Accuracy"),
    ]
    x = np.arange(len(metrics))
    w = 0.20
    for i, (key, _) in enumerate([("old", "fedavg"), ("old", "fedprox"),
                                    ("new", "fedavg"), ("new", "fedprox")]):
        run = (old if key == "old" else new)[f"{_}"]   # noqa: not used
    # build values for each (partition, algo) — 4 series
    series = [
        ("old", "fedavg",  [old["fedavg"][m]  for m, _ in metrics]),
        ("old", "fedprox", [old["fedprox"][m] for m, _ in metrics]),
        ("new", "fedavg",  [new["fedavg"][m]  for m, _ in metrics]),
        ("new", "fedprox", [new["fedprox"][m] for m, _ in metrics]),
    ]
    for i, (part, algo, vals) in enumerate(series):
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w,
                      color=COLORS[(part, algo)], edgecolor="white",
                      label=LABELS[(part, algo)])
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width()/2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Score")
    ax.set_title("(A) Final test metrics — both partitions")
    ax.set_ylim(0, max([v for _, _, vs in series for v in vs]) + 0.10)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")


def panel_per_class(ax, old: dict, new: dict) -> None:
    x = np.arange(len(CLASS_NAMES))
    w = 0.20
    series = [
        ("old", "fedavg",  old["fedavg"]["test_per_class_f1"]),
        ("old", "fedprox", old["fedprox"]["test_per_class_f1"]),
        ("new", "fedavg",  new["fedavg"]["test_per_class_f1"]),
        ("new", "fedprox", new["fedprox"]["test_per_class_f1"]),
    ]
    for i, (part, algo, vals) in enumerate(series):
        offset = (i - 1.5) * w
        ax.bar(x + offset, vals, w,
               color=COLORS[(part, algo)], edgecolor="white",
               label=LABELS[(part, algo)])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_ylabel("Test F1")
    ax.set_title("(B) Per-class test F1 — both partitions")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")


def panel_delta_per_class(ax, old: dict, new: dict) -> None:
    """Δ (FedProx − FedAvg) per class, overlaid for both partitions."""
    deltas_old = np.array(old["fedprox"]["test_per_class_f1"]) - np.array(old["fedavg"]["test_per_class_f1"])
    deltas_new = np.array(new["fedprox"]["test_per_class_f1"]) - np.array(new["fedavg"]["test_per_class_f1"])
    x = np.arange(len(CLASS_NAMES))
    w = 0.40
    ax.bar(x - w/2, deltas_old, w, color="#a0aec0", edgecolor="white",
           label="balanced_specialist (old)")
    ax.bar(x + w/2, deltas_new, w, color="#2b6cb0", edgecolor="white",
           label="balanced_paired (new)")
    ax.axhline(0, color="black", linewidth=0.5)
    for i, d in enumerate(deltas_old):
        ax.annotate(f"{d:+.2f}", xy=(i - w/2, d),
                    xytext=(0, 2 if d >= 0 else -10), textcoords="offset points",
                    ha="center", fontsize=6, color="#4a5568")
    for i, d in enumerate(deltas_new):
        ax.annotate(f"{d:+.2f}", xy=(i + w/2, d),
                    xytext=(0, 2 if d >= 0 else -10), textcoords="offset points",
                    ha="center", fontsize=6, color="black", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_ylabel("Δ F1  (FedProx − FedAvg)")
    ax.set_title("(C) Per-class FedProx advantage — partitions compared")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="mnist_dermnist/results/cpu_flower/flower_comparison_seed42_mu0.01.json")
    ap.add_argument("--new", default="mnist_dermnist/results/cpu_flower_paired/flower_comparison_seed42_mu0.01.json")
    ap.add_argument("--out", default="mnist_dermnist/results/comparison_partitions.png")
    args = ap.parse_args()

    old = load(Path(args.old))
    new = load(Path(args.new))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
    panel_summary_bars(axes[0], old, new)
    panel_per_class(axes[1], old, new)
    panel_delta_per_class(axes[2], old, new)

    # Headline numbers in the suptitle
    fig.suptitle(
        f"FedAvg vs FedProx — partition comparison (single seed, μ=0.01)\n"
        f"Old partition: Δ macro-F1 = {old['delta_macro_f1']:+.4f}    |    "
        f"New partition: Δ macro-F1 = {new['delta_macro_f1']:+.4f}",
        fontsize=12, fontweight="bold", y=1.02,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
