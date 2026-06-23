"""Engineered (balanced_paired_7_clients) partition visualisation.

Sister figure to plot_iid_partition.py. Visualises the per-(client, class)
sample counts under the engineered balanced_paired_7_clients partition.
Each bar is one client; bar segments show the number of samples of each
class. Unlike the IID partition, here client compositions differ
*by design*: each minority class is held by exactly two clients (paired
co-training), while the majority class (mel-nevi) is held by all clients.

Output:
  results/thesis_ready/figures/F_engineered_partition.{pdf,png}

Style: matches F_iid_partition for visual comparison.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fl_dermamnist.data.load import load_dermmnist
from fl_dermamnist.data.partition import balanced_paired_7_clients


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
NPZ_PATH = REPO_ROOT / "dermamnist_64.npz"
OUT_FIG = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

CLASS_COLOURS = [
    "#E76F51",   # actinic
    "#F4A261",   # basal
    "#E9C46A",   # benign keratosis
    "#A8DADC",   # dermatofibroma
    "#2A9D8F",   # melanoma
    "#3D5A80",   # mel-nevi
    "#9C6B9C",   # vascular
]


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
})


def main():
    print(f"Loading DermaMNIST from {NPZ_PATH} ...")
    train, _, _ = load_dermmnist(str(NPZ_PATH), image_size=28)
    labels = np.asarray(train.labels).flatten()
    print(f"  Loaded {len(labels)} training labels")

    client_indices, _ = balanced_paired_7_clients(labels, seed=42)
    K = len(client_indices)
    C = 7

    counts = np.zeros((K, C), dtype=int)
    for k, idx in enumerate(client_indices):
        labels_k = labels[idx]
        for c in range(C):
            counts[k, c] = int((labels_k == c).sum())

    print()
    print(f"{'Client':<8} {'Total':>7} " + " ".join(f"{n[:6]:>7}" for n in CLASS_NAMES))
    for k in range(K):
        print(f"C{k:<7} {counts[k].sum():>7} " + " ".join(f"{counts[k, c]:>7}" for c in range(C)))
    print(f"{'global':<8} {len(labels):>7} " + " ".join(
        f"{int((labels == c).sum()):>7}" for c in range(C)
    ))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(K)
    bottom = np.zeros(K)
    for c in range(C):
        ax.bar(x, counts[:, c], bottom=bottom,
               color=CLASS_COLOURS[c], edgecolor="white", linewidth=0.6,
               label=CLASS_NAMES[c])
        bottom = bottom + counts[:, c]

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k}" for k in range(K)], fontsize=11)
    ax.set_xlabel("Client")
    ax.set_ylabel("Number of training samples")
    ax.set_ylim(0, max(counts.sum(axis=1)) * 1.10)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
        title="Class", fontsize=10, title_fontsize=10,
    )

    for k in range(K):
        total = int(counts[k].sum())
        ax.text(x[k], total + 20, f"$n = {total}$",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_engineered_partition.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
