"""Per-class F1 bar chart for the 2-client 90/10 small-hospital case study.

Reads the single-seed result from
  results/two_client_90_10_rare_stress/test_at_best_*_s42.json
and produces a paired bar chart of FedAvg vs FedProx per-class test F1.
Rare classes (held only by the small Client 1) are highlighted with a
horizontal bracket above their bar pair.

Output:
  results/thesis_ready/figures/F_small_hospital_per_class.{pdf,png}

Single-seed case study; no SEM bands or paired statistics. Caption notes
the n = 1 status.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RUN_DIR = REPO_ROOT / "fl_dermamnist" / "results" / "two_client_90_10_rare_stress"
OUT_FIG = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Actinic\nkeratoses",
    "Basal cell\ncarcinoma",
    "Benign\nkeratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic\nnevi",
    "Vascular\nlesions",
]
# Client 1 (the small site) holds classes {3, 4, 6} = dermato, melanoma, vascular.
SMALL_CLIENT_CLASSES = {3, 4, 6}

COL_FEDAVG  = "#7FBF94"
COL_FEDPROX = "#3D5A80"
HIGHLIGHT   = "#C9A227"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def main():
    fa = json.load(open(RUN_DIR / "test_at_best_fedavg_mu0.0_E20_s42.json"))
    fp = json.load(open(RUN_DIR / "test_at_best_fedprox_mu0.01_E20_s42.json"))

    fa_f1 = np.asarray(fa["per_class_f1"])
    fp_f1 = np.asarray(fp["per_class_f1"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    x = np.arange(len(CLASS_NAMES))
    w = 0.38
    bars_fa = ax.bar(x - w / 2, fa_f1, width=w, color=COL_FEDAVG,
                     label="FedAvg", edgecolor="white", linewidth=0.6)
    bars_fp = ax.bar(x + w / 2, fp_f1, width=w, color=COL_FEDPROX,
                     label=r"FedProx ($\mu = 0.01$)",
                     edgecolor="white", linewidth=0.6)

    # Δ annotation above each pair
    for i, (fa_v, fp_v) in enumerate(zip(fa_f1, fp_f1)):
        d = fp_v - fa_v
        sign = "+" if d >= 0 else ""
        col = "#1f6f3f" if d > 0 else "#b04040" if d < 0 else "#555"
        ax.text(x[i], max(fa_v, fp_v) + 0.04,
                f"$\\Delta = {sign}{d:.3f}$",
                ha="center", va="bottom", fontsize=8.5, color=col)

    # Highlight ONLY the small-client class bars (narrower band so
    # mel-nevi at x=5, held by the large client, stays clearly unmarked).
    for c in SMALL_CLIENT_CLASSES:
        ax.axvspan(x[c] - 0.46, x[c] + 0.46, color=HIGHLIGHT, alpha=0.12,
                   zorder=0)

    ax.set_xticks(x)
    # Mark the small-client classes with a star in the tick label
    tick_labels = [
        f"{name}\n$\\bigstar$" if c in SMALL_CLIENT_CLASSES else name
        for c, name in enumerate(CLASS_NAMES)
    ]
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("Test F1 (seed 42)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(CLASS_NAMES) - 0.5)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend with an extra entry explaining the star + shading.
    star_handle = plt.Line2D([], [], marker="*", color=HIGHLIGHT,
                             linestyle="None", markersize=10,
                             markeredgecolor=HIGHLIGHT,
                             label=r"$\bigstar$ class held only by Client 1 (small site)")
    handles = [bars_fa, bars_fp, star_handle]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize=10, ncol=3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_small_hospital_per_class.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
