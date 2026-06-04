"""Generate F_l1_l4_partition_schematic.pdf — two-panel schematic
comparing L1 and L4 (Methods §4.5.4).

Two panels side by side:
  Panel A — L1 (matched-quantity control): both clients carry every
    class in near-global proportions.
  Panel B — L4 (class-disjoint): Client 0 holds the four common classes,
    Client 1 uniquely holds the three rare classes.

Each panel is a pair of stacked PROPORTIONAL bars (one per client,
sums to 1.0) so the L1 / L4 contrast is immediate visually.
Total client sizes annotated underneath each bar.

All numbers are pulled live from `partition.py`; no hard-coded values.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from mnist_dermnist.data.partition import (
    two_client_90_10_rare_stress,
    two_client_86_14_quantity_only_stratified,
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
CLASS_SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7
RARE = {3, 4, 6}

# Distinct colours per class; rare classes get bright/warm tones, common
# get muted/cool tones so the contrast carries through both panels.
COLOURS = {
    0: "#3b8db5",  # akiec  — blue
    1: "#5a9fc2",  # bcc    — blue
    2: "#7eb3cf",  # bkl    — blue
    3: "#d99c40",  # df     — warm orange (rare)
    4: "#c44e52",  # mel    — red (rare, high-priority)
    5: "#374f6b",  # nv     — dark slate (majority)
    6: "#9b59b6",  # vasc   — purple (rare)
}

# ----------------------------------------------------------------------
# 1. Build allocation matrices from live partition
# ----------------------------------------------------------------------
ROOT_NPZ = ROOT / "dermamnist_64.npz"
train, _, _ = load_dermmnist(str(ROOT_NPZ))
trainy = np.asarray(train.labels)

def alloc_matrix(partitioner, seed=42):
    """Return (2 x 7) matrix of class counts per client."""
    clients, _ = partitioner(train.labels, seed=seed)
    M = np.zeros((2, NUM_CLASSES), dtype=int)
    for cid, idxs in enumerate(clients):
        for c in range(NUM_CLASSES):
            M[cid, c] = int(np.sum(trainy[idxs] == c))
    return M

M_l1 = alloc_matrix(two_client_86_14_quantity_only_stratified)
M_l4 = alloc_matrix(two_client_90_10_rare_stress)

# ----------------------------------------------------------------------
# 2. Plot
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.8, 4.6),
                                gridspec_kw=dict(wspace=0.25))


def draw_panel(ax, M, title_main, title_sub):
    """Draw two proportional stacked bars (C0, C1) for a 2-client matrix."""
    totals = M.sum(axis=1)  # per-client totals
    P = M / totals[:, None]
    x = np.arange(2)
    bottom = np.zeros(2)
    for c in range(NUM_CLASSES):
        ax.bar(x, P[:, c], bottom=bottom,
               color=COLOURS[c], edgecolor="white", linewidth=0.6,
               label=CLASS_SHORT[c] + (" *" if c in RARE else ""))
        # Annotate only with the COUNT (class identity comes from colour)
        for i in range(2):
            if P[i, c] > 0.025:
                colour = "white" if c == 5 else "#1a1a1a"
                ax.text(x[i], bottom[i] + P[i, c] / 2,
                        f"{M[i, c]}",
                        ha="center", va="center", fontsize=8.5,
                        color=colour)
        bottom += P[:, c]
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client 0\n($n = {totals[0]}$)",
                        f"Client 1\n($n = {totals[1]}$)"],
                       fontsize=9.5)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([f"{int(v*100)}%" for v in [0, 0.25, 0.5, 0.75, 1.0]],
                       fontsize=8.5)
    ax.set_ylabel("class proportion within client", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Two-line title: bold main label + italic sub-note on a separate line
    ax.set_title(title_main + "\n" + r"$\it{" + title_sub.replace(" ", "\\ ") + r"}$",
                 fontsize=10.5, pad=10, linespacing=1.5)


draw_panel(axA, M_l1, "L1 — matched-quantity control",
           "quantity skew only")
draw_panel(axB, M_l4, "L4 — class-disjoint mechanism partition",
           "unique rare-class signal on Client 1")

# JS divergence annotation
from scipy.spatial.distance import jensenshannon
def js(M):
    p0 = M[0] / M[0].sum()
    p1 = M[1] / M[1].sum()
    return jensenshannon(p0, p1, base=2) ** 2

axA.text(0.02, -0.20, f"JS divergence $\\approx {js(M_l1):.0e}$",
         transform=axA.transAxes, fontsize=8.5, color="#555")
axB.text(0.02, -0.20, f"JS divergence $= {js(M_l4):.0f}$",
         transform=axB.transAxes, fontsize=8.5, color="#555")

# Shared legend on the right (one row per class, * marks rare)
handles, lbls = axA.get_legend_handles_labels()
fig.legend(handles, lbls, loc="center right",
           bbox_to_anchor=(1.02, 0.5),
           fontsize=8.5, frameon=True, framealpha=0.95,
           title="class\n($*$ = rare)", title_fontsize=8.5)

OUT = ROOT / "mnist_dermnist" / "results" / "thesis_ready" / "figures"
out_pdf = OUT / "F_l1_l4_partition_schematic.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"\nLive verification:")
print(f"  L1 client sizes: {M_l1.sum(axis=1).tolist()} (ratio {M_l1.sum(axis=1)[0]/M_l1.sum():.3f}/{M_l1.sum(axis=1)[1]/M_l1.sum():.3f})")
print(f"  L4 client sizes: {M_l4.sum(axis=1).tolist()} (ratio {M_l4.sum(axis=1)[0]/M_l4.sum():.3f}/{M_l4.sum(axis=1)[1]/M_l4.sum():.3f})")
print(f"  L1 JS = {js(M_l1):.2e}")
print(f"  L4 JS = {js(M_l4):.4f}")
print(f"  L4 C0 nonzero classes: {[c for c in range(7) if M_l4[0,c]>0]}")
print(f"  L4 C1 nonzero classes: {[c for c in range(7) if M_l4[1,c]>0]}")
plt.close(fig)
