"""Two-panel figure for the two new partitions introduced in s2.5.1.

Side-by-side stacked-bar comparison of:
  (a) Dirichlet alpha = 0.1          -- dirichlet_7_clients(alpha=0.1)
  (b) Specialist (1-of-7)            -- specialist_7_clients

The IID and engineered (balanced-paired) partitions are NOT redrawn here
because they already have dedicated figures earlier in the chapter
(F_iid_partition, F_engineered_partition). This figure introduces only
the two new partition designs used in the robustness section.

Panels share the same class palette as F_iid_partition /
F_engineered_partition so cross-section visual comparison is direct.
Per-panel y-axes are independent (Dirichlet has wildly different
per-client totals: 91 to 3,721 in the seed-42 realisation). Each bar is
annotated with n_k above it so quantity skew is explicit rather than
hidden.

Output:
  results/thesis_ready/figures/F_partition_comparison_dir_spec.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fl_dermamnist.data.load import load_dermmnist
from fl_dermamnist.data.partition import (
    dirichlet_7_clients,
    specialist_7_clients,
)


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
    "axes.titlesize":    11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
})


def counts_matrix(client_indices, labels, C=7):
    K = len(client_indices)
    M = np.zeros((K, C), dtype=int)
    for k, idx in enumerate(client_indices):
        labels_k = labels[idx]
        for c in range(C):
            M[k, c] = int((labels_k == c).sum())
    return M


def panel(ax, M, title, K_labels=None):
    K, C = M.shape
    x = np.arange(K)
    bottom = np.zeros(K)
    for c in range(C):
        ax.bar(x, M[:, c], bottom=bottom,
               color=CLASS_COLOURS[c], edgecolor="white", linewidth=0.5,
               label=CLASS_NAMES[c])
        bottom = bottom + M[:, c]

    ax.set_xticks(x)
    if K_labels is None:
        K_labels = [f"C{k}" for k in range(K)]
    ax.set_xticklabels(K_labels, fontsize=10)
    y_max = max(M.sum(axis=1)) * 1.15
    ax.set_ylim(0, y_max)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)

    for k in range(K):
        total = int(M[k].sum())
        ax.text(x[k], total + y_max * 0.015, f"{total}",
                ha="center", va="bottom", fontsize=8)


def main():
    print(f"Loading DermaMNIST from {NPZ_PATH} ...")
    train, _, _ = load_dermmnist(str(NPZ_PATH), image_size=28)
    labels = np.asarray(train.labels).flatten()
    print(f"  Loaded {len(labels)} training labels")

    partitions = [
        ("(a) Dirichlet $\\alpha = 0.1$",        dirichlet_7_clients(labels, seed=42, alpha=0.1)[0]),
        ("(b) Specialist (1-of-7)",              specialist_7_clients(labels, seed=42)[0]),
    ]
    Ms = [counts_matrix(c, labels) for _, c in partitions]

    for (title, _), M in zip(partitions, Ms):
        totals = M.sum(axis=1)
        print(f"{title:<42} totals: {totals.tolist()}  range=[{totals.min()}, {totals.max()}]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                             gridspec_kw={"wspace": 0.18})
    for ax, (title, _), M in zip(axes, partitions, Ms):
        panel(ax, M, title)

    axes[0].set_ylabel("Number of training samples")
    axes[0].set_xlabel("Client")
    axes[1].set_xlabel("Client")

    # Single shared legend on the right.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[c], label=CLASS_NAMES[c])
        for c in range(7)
    ]
    fig.legend(
        handles=handles,
        loc="center left", bbox_to_anchor=(0.92, 0.5),
        frameon=False, title="Class", fontsize=10, title_fontsize=10,
    )

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_partition_comparison_dir_spec.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
