"""Single-seed deep-dive visualisation for the best FedProx seed -- F9.

For the Flower runtime engineered partition, the seed with the highest
absolute FedProx test macro-F1 is 161803 (FedProx = 0.5374,
FedAvg = 0.5069, paired Delta = +0.0306).

Layout: 2x4 grid
  Panel 1 = global val_macro_f1 over rounds (FedAvg vs FedProx)
  Panels 2-8 = per-class val_f1_class_<c> over rounds for the 7 classes

Each panel shows the single-seed trajectory (no error band -- it's one seed).
Annotated with the test-set F1 at best-val checkpoint for both algorithms
and their paired Delta.

Output:
  results/thesis_ready/figures/F9_best_seed_fedprox.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS_ROOT = REPO_ROOT / "fl_dermamnist" / "results"
SWEEP = RESULTS_ROOT / "flower_C0_baseline"
OUT_FIG = RESULTS_ROOT / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150
BEST_SEED = 161803

CLASS_DISPLAY = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]
CLASS_PREVALENCE = [0.0327, 0.0513, 0.1097, 0.0115, 0.1111, 0.6705, 0.0141]

COL_FEDAVG  = "#7FBF94"   # soft mint green
COL_FEDPROX = "#3D5A80"   # dark slate blue


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def main():
    print(f"Single-seed deep-dive for seed {BEST_SEED} (best FedProx)")
    fa_hist_p = SWEEP / f"history_fedavg_mu0.0_E20_s{BEST_SEED}.csv"
    fp_hist_p = SWEEP / f"history_fedprox_mu0.01_E20_s{BEST_SEED}.csv"
    fa_test_p = SWEEP / f"test_at_best_fedavg_mu0.0_E20_s{BEST_SEED}.json"
    fp_test_p = SWEEP / f"test_at_best_fedprox_mu0.01_E20_s{BEST_SEED}.json"

    fa_hist = pd.read_csv(fa_hist_p)
    fp_hist = pd.read_csv(fp_hist_p)
    fa_test = json.load(open(fa_test_p))
    fp_test = json.load(open(fp_test_p))

    print(f"FedAvg  test macro-F1: {fa_test['macro_f1']:.4f} "
          f"(at round {fa_test['selected_round']})")
    print(f"FedProx test macro-F1: {fp_test['macro_f1']:.4f} "
          f"(at round {fp_test['selected_round']})")
    print(f"Paired Δ:              {fp_test['macro_f1'] - fa_test['macro_f1']:+.4f}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # ------- Panel 0: Global macro-F1 ----------------------------------------
    ax = axes[0]
    ax.plot(fa_hist["round"], fa_hist["val_macro_f1"], color=COL_FEDAVG,
            linewidth=1.5, label="FedAvg")
    ax.plot(fp_hist["round"], fp_hist["val_macro_f1"], color=COL_FEDPROX,
            linewidth=1.5, label="FedProx ($\\mu=0.01$)")
    # Mark best-val checkpoint with a vertical line
    ax.axvline(fa_test["selected_round"], color=COL_FEDAVG,
               linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(fp_test["selected_round"], color=COL_FEDPROX,
               linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Global macro-F1 (validation)",
                 loc="left", fontweight="bold", pad=6, fontsize=10)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1")
    ax.set_xlim(0, NUM_ROUNDS)
    ax.set_ylim(0, 0.65)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    # ------- Panels 1-7: Per-class val_f1 ------------------------------------
    for c in range(7):
        ax = axes[c + 1]
        ycol = f"val_f1_class_{c}"
        ax.plot(fa_hist["round"], fa_hist[ycol], color=COL_FEDAVG,
                linewidth=1.4, label="FedAvg")
        ax.plot(fp_hist["round"], fp_hist[ycol], color=COL_FEDPROX,
                linewidth=1.4, label="FedProx")
        title = f"{CLASS_DISPLAY[c]} ({CLASS_PREVALENCE[c] * 100:.1f}\\%)"
        ax.set_title(title, loc="left", fontweight="bold", pad=6, fontsize=9)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Validation F1")
        ax.set_xlim(0, NUM_ROUNDS)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    fig.suptitle(
        f"Single-seed deep-dive: seed {BEST_SEED} (FedProx's best seed by absolute F1; "
        f"Flower runtime, engineered partition)",
        fontsize=12, y=1.01, fontweight="bold"
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F9_best_seed_fedprox.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
