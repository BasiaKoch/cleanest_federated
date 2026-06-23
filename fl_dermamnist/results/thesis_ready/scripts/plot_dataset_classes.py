"""DermaMNIST dataset visualisation -- minimal one-image-per-class.

Single 1x7 row showing one representative training sample for each of the
seven DermaMNIST classes. Class name shown below each image.
Inspired by the visualisation in Kapo et al. (2024), "Super-Resolution
of DermaMNIST Images Using Deep Learning Models", IEEE.

Output:
  results/thesis_ready/figures/F_dataset_classes.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
NPZ_PATH = REPO_ROOT / "dermamnist_64.npz"
OUT_FIG = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Actinic\nkeratoses",
    "Basal cell\ncarcinoma",
    "Benign\nkeratosis",
    "Dermato-\nfibroma",
    "Melanoma",
    "Melanocytic\nnevi",
    "Vascular\nlesions",
]
SAMPLE_SEED = 42


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RARE = {3, 4, 6}      # dermatofibroma, melanoma, vascular (the rare-class group)
NV = 5                # melanocytic nevi -- the majority attractor
ORANGE, NV_COL, OTHER = "#E67E22", "#34495E", "#AEB6BF"


def main():
    print(f"Loading DermaMNIST from {NPZ_PATH} ...")
    data = np.load(NPZ_PATH)
    images = data["train_images"]
    labels = np.asarray(data["train_labels"]).flatten()

    # real per-class training prevalence (computed from the labels, not hard-coded)
    counts = np.bincount(labels, minlength=7)
    prev = counts / counts.sum() * 100.0

    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = [rng.choice(np.where(labels == c)[0]) for c in range(7)]
    bar_colors = [ORANGE if c in RARE else (NV_COL if c == NV else OTHER)
                  for c in range(7)]

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 7, height_ratios=[2.3, 2.1], hspace=0.55, wspace=0.06)

    # --- top: one representative image per class ---
    for c in range(7):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(images[chosen[c]])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#666666")
        ax.set_xlabel(CLASS_NAMES[c], fontsize=9, labelpad=3)

    # --- bottom: training-set class prevalence (makes the imbalance explicit) ---
    axb = fig.add_subplot(gs[1, :])
    bars = axb.bar(range(7), prev, color=bar_colors, edgecolor="#333333",
                   linewidth=0.6, width=0.74)
    axb.set_xticks(range(7))
    axb.set_xticklabels(SHORT, fontsize=9)
    axb.set_ylabel("Train prevalence (%)", fontsize=10)
    axb.set_ylim(0, 74)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    for c, b in enumerate(bars):
        # truncate to 1 dp to match the thesis-canonical figures (e.g. nv 66.9%)
        pct = np.floor(prev[c] * 10) / 10
        axb.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.3,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.5,
                 fontweight=("bold" if (c in RARE or c == NV) else "normal"),
                 color=("#B9521E" if c in RARE else "#222222"))
    axb.text(0.985, 0.90, "rare classes (orange)", transform=axb.transAxes,
             ha="right", va="top", fontsize=8.5, color=ORANGE, fontweight="bold")

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_dataset_classes.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
