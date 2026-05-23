"""Partition-design visualisation — F5.

Produces a single figure showing how each of the 4 statistical-heterogeneity
partition designs distributes the 7 DermaMNIST classes across the 7 clients.

This figure is referenced by the §Results "Statistical heterogeneity"
section as the motivation for each partition design.

Layout: 2x2 grid of class-distribution heatmaps.
  (a) IID partition           — control / theoretical null
  (b) Dirichlet alpha=0.1     — literature-standard non-IID
  (c) Specialist (1-of-7)     — singleton ownership; pairing-lever check
  (d) Balanced paired (engineered) — every minority class held by exactly two clients

Each panel:
  Rows    = 7 clients (C0 ... C6)
  Columns = 7 DermaMNIST classes
  Cell    = number of training samples of that class held by that client
  Annotation: integer count if > 0
  Colour: log-scale viridis (handles the 100x dynamic range cleanly)

Output:
  results/thesis_ready/figures/F5_partition_distributions.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_dermnist.data.load import load_dermmnist
from mnist_dermnist.data.partition import (
    balanced_paired_7_clients,
    dirichlet_7_clients,
    iid_7_clients,
    specialist_7_clients,
)


# ----- Paths --------------------------------------------------------------

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS = REPO_ROOT / "mnist_dermnist" / "results"
OUT_FIG = RESULTS / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
NPZ = REPO_ROOT / "dermamnist_64.npz"

# Class labels per DermaMNIST canonical order
CLASS_DISPLAY = [
    "Actinic\nker.",
    "Basal\ncell\ncarc.",
    "Benign\nker.",
    "Dermato-\nfibroma",
    "Melanoma",
    "Melanocytic\nnevi",
    "Vascular\nles.",
]
N_CLASSES = 7
N_CLIENTS = 7


# ----- Style ---------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
})


def _dirichlet_a01(labels, seed=42):
    return dirichlet_7_clients(labels, seed=seed, alpha=0.1)


# ----- Helpers -------------------------------------------------------------

def compute_count_matrix(client_indices: list[list[int]], labels: np.ndarray) -> np.ndarray:
    """Build a (n_clients, n_classes) integer count matrix."""
    M = np.zeros((N_CLIENTS, N_CLASSES), dtype=int)
    for c, idx in enumerate(client_indices):
        sub = labels[idx]
        for k in range(N_CLASSES):
            M[c, k] = int((sub == k).sum())
    return M


def _annotate_heatmap(ax, M):
    """Annotate each cell with its integer count (skip zeros)."""
    n_rows, n_cols = M.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = int(M[i, j])
            if v == 0:
                continue
            # contrast colour: light text on dark cells, dark text on light cells
            log_v_norm = (np.log1p(v) - np.log1p(M.min())) / max(
                np.log1p(M.max()) - np.log1p(M.min()), 1e-9
            )
            txt_color = "white" if log_v_norm > 0.55 else "black"
            ax.text(j, i, str(v),
                    ha="center", va="center", color=txt_color, fontsize=8)


def _imshow_log(ax, M, *, vmax=None):
    """Heatmap on log(1+count) scale, viridis."""
    vmax = vmax or M.max()
    im = ax.imshow(np.log1p(M), aspect="auto", cmap="viridis",
                   vmin=0, vmax=np.log1p(vmax), interpolation="nearest")
    return im


# ----- Main figure ---------------------------------------------------------

def main():
    print(f"Loading DermaMNIST from {NPZ} ...")
    train, val, test = load_dermmnist(str(NPZ), image_size=28)
    labels = np.asarray(train.labels).flatten()
    print(f"Train labels: shape={labels.shape}, classes={sorted(set(labels.tolist()))}")
    print(f"Class counts: {np.bincount(labels, minlength=7).tolist()}")

    # Compute count matrices for each partition (use seed=42 for representative)
    partitions = [
        ("(a) IID — uniform random",                  iid_7_clients),
        (r"(b) Dirichlet $\alpha=0.1$ — literature-standard non-IID", _dirichlet_a01),
        ("(c) Specialist — 1 minority class per client",  specialist_7_clients),
        ("(d) Balanced paired — engineered (every minority\n     class held by exactly 2 clients)",
                                                       balanced_paired_7_clients),
    ]

    # Compute global vmax for colour-scale consistency
    matrices = []
    for title, fn in partitions:
        try:
            client_indices, _ = fn(labels, seed=42)
        except Exception as e:
            print(f"FAILED to compute partition {title}: {e}")
            matrices.append((title, None))
            continue
        M = compute_count_matrix(client_indices, labels)
        matrices.append((title, M))
        print(f"\n{title}")
        print(f"   shape={M.shape}, total samples={M.sum()}, "
              f"per-client {M.sum(axis=1).tolist()}")

    global_vmax = max(M.max() for _, M in matrices if M is not None)

    # ----- Plot ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (title, M) in zip(axes, matrices):
        if M is None:
            ax.set_title(title + " — (compute failed)")
            ax.axis("off")
            continue
        im = _imshow_log(ax, M, vmax=global_vmax)
        _annotate_heatmap(ax, M)
        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_DISPLAY, fontsize=8)
        # Embed per-client total in y-tick label so it doesn't overflow
        totals = M.sum(axis=1)
        ax.set_yticks(range(N_CLIENTS))
        ax.set_yticklabels([f"C{i}  (n={t})" for i, t in enumerate(totals)],
                           fontsize=9)
        ax.set_title(title, loc="left", fontweight="bold", pad=8)

    # Add shared colourbar on the right (log scale ticks)
    fig.subplots_adjust(left=0.07, right=0.91, top=0.92, bottom=0.07,
                        wspace=0.30, hspace=0.30)
    cbar_ax = fig.add_axes([0.93, 0.10, 0.015, 0.78])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Samples per (client, class)\n[$\\log(1+\\mathrm{count})$ colour scale]",
                   fontsize=9)

    fig.suptitle("Class distribution of training samples across the 7 clients, "
                 "under each statistical-heterogeneity partition design",
                 fontsize=12, y=0.98, fontweight="bold")
    # Add caption about what to look for
    caption = (
        "Each row is one client (C0–C6) and each column is one of the 7 DermaMNIST classes. "
        "Cell colour shows the number of training samples on a $\\log(1+\\mathrm{count})$ scale; "
        "integer counts are annotated inside each cell. Per-client total $n$ is shown in the "
        "row label. The IID partition is uniform; Dirichlet $\\alpha=0.1$ produces severe label + "
        "quantity skew; the specialist partition assigns each minority class to exactly one client; "
        "the engineered balanced-paired partition assigns each minority class to exactly two clients "
        "with the majority class (melanocytic nevi) split across all seven clients."
    )
    fig.text(0.5, 0.005, caption, ha="center", fontsize=9, style="italic",
             wrap=True)

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F5_partition_distributions.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
