"""Single-seed Colab result: convergence curves + final-test comparison.

Builds a 6-panel figure summarising a paired-seed run when we only have
the FedProx per-round CSV (FedAvg's history may be lost — only its final
test metrics remain in the comparison JSON).

Panels:
  (A) FedProx val_macro_f1 curve with FedAvg test-macro-F1 reference line
  (B) FedProx val_loss curve
  (C) FedProx train_loss (log scale) — shows whether training collapsed
  (D) Final test metrics — FedAvg vs FedProx grouped bars
  (E) Per-class test F1 — FedAvg vs FedProx grouped bars
  (F) Per-class Δ (FedProx − FedAvg) signed bars

Usage:
    PYTHONPATH=. python -m mnist_dermnist.analysis.plot_single_seed_report \\
        --json mnist_dermnist/results/colab_recovered/flower_comparison_seed42_mu0.01.json \\
        --fedprox-csv mnist_dermnist/results/colab_recovered/history_fedprox_mu0.01_E20_s42.csv \\
        --out mnist_dermnist/results/colab_recovered/single_seed_report_s42.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_FA = "#2b6cb0"   # blue
COLOR_FP = "#dd6b20"   # orange
CLASS_NAMES = ["actinic", "basal", "benign\nkerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--fedprox-csv", required=True)
    ap.add_argument("--fedavg-csv", default=None,
                    help="Optional: also plot FedAvg per-round curves if available")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.json))
    fa = d["fedavg"]; fp = d["fedprox"]
    fp_curve = pd.read_csv(args.fedprox_csv)
    fa_curve = pd.read_csv(args.fedavg_csv) if args.fedavg_csv else None

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # ---------- (A) val_macro_f1 curve ----------
    ax = axes[0, 0]
    ax.plot(fp_curve["round"], fp_curve["val_macro_f1"],
            color=COLOR_FP, lw=2, label="FedProx (val)")
    if fa_curve is not None:
        ax.plot(fa_curve["round"], fa_curve["val_macro_f1"],
                color=COLOR_FA, lw=2, label="FedAvg (val)")
    # Mark best-val rounds
    fp_best_idx = fp_curve["val_macro_f1"].idxmax()
    ax.scatter([fp_curve.loc[fp_best_idx, "round"]],
               [fp_curve.loc[fp_best_idx, "val_macro_f1"]],
               s=70, color=COLOR_FP, edgecolor="white", zorder=5,
               label=f"FedProx best val (r={int(fp_curve.loc[fp_best_idx, 'round'])})")
    ax.axhline(fp["test_macro_f1"], color=COLOR_FP, ls=":", alpha=0.6,
               label=f"FedProx test = {fp['test_macro_f1']:.3f}")
    ax.axhline(fa["test_macro_f1"], color=COLOR_FA, ls=":", alpha=0.6,
               label=f"FedAvg test = {fa['test_macro_f1']:.3f}")
    ax.set_xlabel("Round"); ax.set_ylabel("Macro-F1")
    ax.set_title("(A) Validation macro-F1 vs round")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # ---------- (B) val_loss curve ----------
    ax = axes[0, 1]
    ax.plot(fp_curve["round"], fp_curve["val_loss"],
            color=COLOR_FP, lw=2, label="FedProx")
    if fa_curve is not None:
        ax.plot(fa_curve["round"], fa_curve["val_loss"],
                color=COLOR_FA, lw=2, label="FedAvg")
    ax.set_xlabel("Round"); ax.set_ylabel("Cross-entropy")
    ax.set_title("(B) Validation loss vs round")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---------- (C) train_loss log scale ----------
    ax = axes[0, 2]
    ax.plot(fp_curve["round"], fp_curve["train_loss_weighted"],
            color=COLOR_FP, lw=2, label="FedProx")
    if fa_curve is not None:
        ax.plot(fa_curve["round"], fa_curve["train_loss_weighted"],
                color=COLOR_FA, lw=2, label="FedAvg")
    ax.set_yscale("log")
    ax.set_xlabel("Round"); ax.set_ylabel("Train loss (log)")
    ax.set_title("(C) Training loss (size-weighted, log scale)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # ---------- (D) Final test metrics bars ----------
    ax = axes[1, 0]
    metrics  = [("test_macro_f1", "Macro-F1"),
                ("test_balanced_accuracy", "Balanced acc."),
                ("test_accuracy", "Accuracy")]
    x = np.arange(len(metrics)); w = 0.36
    fa_vals = [fa[k] for k, _ in metrics]
    fp_vals = [fp[k] for k, _ in metrics]
    ax.bar(x - w/2, fa_vals, w, color=COLOR_FA, edgecolor="white", label="FedAvg")
    ax.bar(x + w/2, fp_vals, w, color=COLOR_FP, edgecolor="white", label="FedProx")
    for i, (a, p) in enumerate(zip(fa_vals, fp_vals)):
        ax.annotate(f"{a:.3f}", (i - w/2, a), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{p:.3f}", (i + w/2, p), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("(D) Final test metrics")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # ---------- (E) Per-class F1 grouped ----------
    ax = axes[1, 1]
    x = np.arange(len(CLASS_NAMES)); w = 0.36
    ax.bar(x - w/2, fa["test_per_class_f1"], w,
           color=COLOR_FA, edgecolor="white", label="FedAvg")
    ax.bar(x + w/2, fp["test_per_class_f1"], w,
           color=COLOR_FP, edgecolor="white", label="FedProx")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_ylabel("Test F1"); ax.set_ylim(0, 1.0)
    ax.set_title("(E) Per-class test F1")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # ---------- (F) Per-class delta ----------
    ax = axes[1, 2]
    deltas = np.array(fp["test_per_class_f1"]) - np.array(fa["test_per_class_f1"])
    colors = ["#38a169" if d >= 0 else "#e53e3e" for d in deltas]
    bars = ax.bar(x, deltas, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(0.05, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axhline(-0.05, color="gray", lw=0.5, ls="--", alpha=0.5,
               label="±0.05 tolerance")
    for bar, dv in zip(bars, deltas):
        ax.annotate(f"{dv:+.3f}",
                    (bar.get_x() + bar.get_width()/2, dv),
                    xytext=(0, 3 if dv >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8,
                    fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_ylabel("Δ F1 (FedProx − FedAvg)")
    ax.set_title("(F) Per-class Δ (green = FedProx better)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")

    # ---------- Super title ----------
    seed = d.get("seed", "?"); mu = d.get("mu", "?")
    E = d.get("local_epochs", "?"); R = d.get("num_rounds", "?")
    part = d.get("partition", "?")
    delta_f1 = d.get("delta_macro_f1", fp["test_macro_f1"] - fa["test_macro_f1"])
    fig.suptitle(
        f"FedAvg vs FedProx — single seed={seed}, μ={mu}, E={E}, R={R}, partition={part}\n"
        f"Δ test macro-F1 = {delta_f1:+.4f}    Δ balanced acc = {d.get('delta_balanced_accuracy', 0):+.4f}",
        fontsize=12, fontweight="bold"
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
