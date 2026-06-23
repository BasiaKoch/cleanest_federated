"""DermaMNIST dataset visualisation -- example images + training-set prevalence.

Two-part Methods figure that motivates macro-F1 over accuracy:
  (top)    one representative training image per class, framed by group
  (bottom) training-set class prevalence (computed from the labels), with the
           rare-class group {dermatofibroma, melanoma, vascular} highlighted and
           the majority class (melanocytic nevi) marked as the majority attractor.

Prevalences are computed from the labels (not hard-coded) and shown truncated to
one decimal place to match the thesis-canonical values (e.g. nv 66.9%).

Output: results/thesis_ready/figures/F_dataset_classes.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from fl_dermamnist.common.paths import repo_root

REPO_ROOT = repo_root()
NPZ_PATH = REPO_ROOT / "dermamnist_64.npz"
OUT_FIG = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

FULL = ["Actinic keratoses", "Basal cell carcinoma", "Benign keratosis",
        "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions"]
SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RARE = {3, 4, 6}      # dermatofibroma, melanoma, vascular -- the rare-class group
NV = 5                # melanocytic nevi -- the majority attractor
SAMPLE_SEED = 42

# restrained palette: amber (rare group) | slate (majority) | grey (other)
AMBER, AMBER_DK = "#E67E22", "#B9521E"
SLATE = "#2C3E50"
GREY = "#B8C0C8"
INK = "#222222"
NOTE = "#4d555e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def group_color(c):
    return AMBER if c in RARE else (SLATE if c == NV else GREY)


def main():
    print(f"Loading DermaMNIST from {NPZ_PATH} ...")
    data = np.load(NPZ_PATH)
    images = data["train_images"]
    labels = np.asarray(data["train_labels"]).flatten()

    counts = np.bincount(labels, minlength=7)
    prev = counts / counts.sum() * 100.0
    pct = np.floor(prev * 10) / 10.0     # 1-dp truncation (keeps nv = 66.9%, etc.)

    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = [rng.choice(np.where(labels == c)[0]) for c in range(7)]

    fig = plt.figure(figsize=(12.0, 4.9))
    gs = fig.add_gridspec(2, 7, height_ratios=[2.0, 2.6], hspace=0.52, wspace=0.14)

    # --- top: one representative image per class, framed by its group --------
    for c in range(7):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(images[chosen[c]])
        ax.set_xticks([]); ax.set_yticks([])
        gc = group_color(c)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(gc)
            sp.set_linewidth(2.4 if (c in RARE or c == NV) else 1.0)
        ax.text(0.5, -0.11, SHORT[c], transform=ax.transAxes, ha="center", va="top",
                fontsize=11.5, fontweight="bold", color=gc)
        ax.text(0.5, -0.32, FULL[c], transform=ax.transAxes, ha="center", va="top",
                fontsize=8.2, color=NOTE)

    # --- bottom: training-set class prevalence -------------------------------
    axb = fig.add_subplot(gs[1, :])
    bars = axb.bar(range(7), prev, color=[group_color(c) for c in range(7)],
                   edgecolor="#333333", linewidth=0.6, width=0.72)
    axb.set_xticks(range(7)); axb.set_xticklabels(SHORT, fontsize=9.5)
    axb.set_ylabel("Training-set prevalence (%)", fontsize=10)
    axb.set_ylim(0, 78)
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)

    for c, b in enumerate(bars):
        x = b.get_x() + b.get_width() / 2
        if c == NV:
            axb.text(x, b.get_height() + 1.3, f"{pct[c]:.1f}%\nmajority class",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                     color=SLATE, linespacing=1.2)
        else:
            axb.text(x, b.get_height() + 1.3, f"{pct[c]:.1f}%", ha="center",
                     va="bottom", fontsize=8.6,
                     fontweight=("bold" if c in RARE else "normal"),
                     color=(AMBER_DK if c in RARE else INK))

    # restrained legend: the three groups
    handles = [mpatches.Patch(fc=AMBER, ec="#333333", lw=0.5, label="rare-class group: df, mel, vasc"),
               mpatches.Patch(fc=SLATE, ec="#333333", lw=0.5, label="majority class: nv"),
               mpatches.Patch(fc=GREY, ec="#333333", lw=0.5, label="other classes")]
    axb.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.97),
               frameon=False, fontsize=8.6, handlelength=1.1, labelspacing=0.45,
               borderpad=0.2)

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_dataset_classes.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
