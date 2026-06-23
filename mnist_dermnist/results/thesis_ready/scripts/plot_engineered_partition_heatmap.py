"""Generate F_engineered_partition_heatmap.pdf — class-by-client heatmap
for the engineered balanced-paired 7-client partition (§4.5 Methods).

Builds the heatmap directly from the partition spec in
mnist_dermnist/data/partition.py so the figure is repo-grounded.

Layout:
  rows    = 7 DermaMNIST classes (with full names)
  columns = Client 0 to Client 6
  cells   = training-sample counts (log-scale colour so minority cells
            are still visible despite mel-nevi's 670-per-client mass)
  footer  = per-client total training samples
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Make the package importable when run directly, regardless of this file's
# depth (walk up to the directory that contains the mnist_dermnist package).
_HERE = Path(__file__).resolve()
for _cand in _HERE.parents:
    if (_cand / "mnist_dermnist" / "__init__.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mnist_dermnist.common.paths import repo_root, thesis_figures_dir
from mnist_dermnist.data.partition import (
    BALANCED_PAIRED_7_CLIENTS_SPEC,
    balanced_paired_7_clients,
)
from mnist_dermnist.data.load import load_dermmnist


CLASS_LONG = [
    "actinic keratoses",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "mel-nevi (majority)",
    "vascular lesions",
]
NUM_CLASSES = 7
NUM_CLIENTS = 7

# ----------------------------------------------------------------------
# 1. Build allocation matrix straight from the spec
# ----------------------------------------------------------------------
M = np.zeros((NUM_CLASSES, NUM_CLIENTS), dtype=int)
for entry in BALANCED_PAIRED_7_CLIENTS_SPEC:
    cid = entry["id"]
    for c, n in entry["per_class"].items():
        M[c, cid] = n
per_client_total = M.sum(axis=0)

# Cross-check against the live partition
ROOT_NPZ = repo_root() / "dermamnist_64.npz"
if ROOT_NPZ.exists():
    train, _, _ = load_dermmnist(str(ROOT_NPZ))
    clients_idx, _ = balanced_paired_7_clients(train.labels, seed=42)
    live_sizes = [len(s) for s in clients_idx]
    assert per_client_total.tolist() == live_sizes, \
        f"spec/live size mismatch: spec={per_client_total.tolist()} vs live={live_sizes}"

# ----------------------------------------------------------------------
# 2. Plot
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.size": 9.5})

fig, (ax, ax_tot) = plt.subplots(
    2, 1, figsize=(6.6, 4.4),
    gridspec_kw=dict(height_ratios=[7, 1], hspace=0.08),
)

# Mask zeros so log-norm doesn't choke; show empty cells as white
masked = np.ma.masked_equal(M, 0)
cmap = plt.cm.YlGnBu
cmap.set_bad(color="#f7f7f7")  # near-white for empty cells

im = ax.imshow(
    masked, aspect="auto", cmap=cmap,
    norm=LogNorm(vmin=1, vmax=M.max()),
)

# Annotate non-zero cells with the count
for c in range(NUM_CLASSES):
    for cid in range(NUM_CLIENTS):
        v = M[c, cid]
        if v > 0:
            # White text on dark cells, black on light
            colour = "white" if v >= 400 else "#222"
            ax.text(cid, c, f"{v}", ha="center", va="center",
                    fontsize=8.5, color=colour)

ax.set_xticks(range(NUM_CLIENTS))
ax.set_xticklabels([f"C{i}" for i in range(NUM_CLIENTS)])
ax.set_yticks(range(NUM_CLASSES))
ax.set_yticklabels(CLASS_LONG, fontsize=9)
ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False, top=False)
ax.tick_params(axis="y", which="both", left=False)

# Red outline around each rare-class row (df=3, mel=4, vasc=6)
for r in (3, 4, 6):
    ax.add_patch(plt.Rectangle((-0.5, r - 0.5), NUM_CLIENTS, 1,
                                fill=False, edgecolor="#C44E52",
                                linewidth=1.2, zorder=5))

# Colour bar on the right
cbar = fig.colorbar(im, ax=[ax, ax_tot], shrink=0.65, pad=0.02)
cbar.set_label("training samples per cell\n(log scale)", fontsize=8.5)
cbar.ax.tick_params(labelsize=8)

# Bottom strip: per-client totals as a single-row bar
ax_tot.bar(range(NUM_CLIENTS), per_client_total,
           color="#5b8db5", edgecolor="black", linewidth=0.5, width=0.85)
for cid, tot in enumerate(per_client_total):
    ax_tot.text(cid, tot + 50, f"{tot}", ha="center", va="bottom",
                fontsize=8.5)
ax_tot.set_xticks(range(NUM_CLIENTS))
ax_tot.set_xticklabels([f"C{i}" for i in range(NUM_CLIENTS)], fontsize=9)
ax_tot.set_ylim(0, max(per_client_total) * 1.18)
ax_tot.set_ylabel("client size", fontsize=9)
ax_tot.spines["top"].set_visible(False)
ax_tot.spines["right"].set_visible(False)
ax_tot.tick_params(axis="x", which="both", top=False)
ax_tot.tick_params(axis="y", labelsize=8)
ax_tot.grid(axis="y", linestyle=":", alpha=0.4)

# Title (compact)
ax.set_title(
    "Engineered balanced-paired 7-client partition: "
    "class $\\times$ client allocation",
    fontsize=10.5, pad=8,
)

OUT = thesis_figures_dir()
out_pdf = OUT / "F_engineered_partition_heatmap.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_pdf}")
plt.close(fig)
