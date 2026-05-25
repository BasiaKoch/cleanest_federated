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
OUT_FIG = REPO_ROOT / "mnist_dermnist" / "results" / "thesis_ready" / "figures"
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


def main():
    print(f"Loading DermaMNIST from {NPZ_PATH} ...")
    data = np.load(NPZ_PATH)
    images = data["train_images"]
    labels = np.asarray(data["train_labels"]).flatten()

    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = [rng.choice(np.where(labels == c)[0]) for c in range(7)]

    fig, axes = plt.subplots(1, 7, figsize=(11, 2.2),
                             gridspec_kw={"wspace": 0.05})
    for c in range(7):
        ax = axes[c]
        ax.imshow(images[chosen[c]])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#666666")
        ax.set_xlabel(CLASS_NAMES[c], fontsize=10, labelpad=4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_dataset_classes.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
