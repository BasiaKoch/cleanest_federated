"""Generate F_l1_l4_partition_schematic.pdf -- two-panel L1 vs L4 schematic
(Methods, mechanism design).

The negative-control logic in one figure: BOTH partitions impose the same
86 % / 14 % client-size skew; only the class structure differs.
  Panel A -- L1 (quantity-only control): both clients carry a near-global class
    mix (JS approx 0).
  Panel B -- L4 (class-disjoint): Client 0 holds the common classes + majority,
    Client 1 uniquely holds the rare-class set {df, mel, vasc} (JS = 1).

Each panel is a pair of proportional stacked bars (one per client, sums to
100 %), with exact per-class counts annotated. All numbers are computed live
from the partitioners in partition.py (no hard-coded values); only the label
array is needed, so the figure regenerates without the full training env.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import to_rgba

# Make the package importable when run directly.
_HERE = Path(__file__).resolve()
for _cand in _HERE.parents:
    if (_cand / "fl_dermamnist" / "__init__.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from fl_dermamnist.common.paths import repo_root, thesis_figures_dir
from fl_dermamnist.data.partition import (
    two_client_90_10_rare_stress,
    two_client_86_14_quantity_only_stratified,
)


CLASS_SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7
RARE = {3, 4, 6}          # df, mel, vasc -- the rare-class set
NV = 5                    # mel-nevi -- the majority

# Palette consistent with Fig 2.2: common = greys, rare set = ambers, nv = slate
INK, MUTE = "#1b1b1b", "#56606b"
AMBER = "#E2853A"
COLOURS = {
    0: "#C2CBD3",  # akiec  common (light grey)
    1: "#93A1AE",  # bcc    common
    2: "#67768A",  # bkl    common (dark grey)
    3: "#F1C57C",  # df     rare (light amber)
    4: "#E2853A",  # mel    rare (amber)
    5: "#2C3E50",  # nv     majority (slate)
    6: "#AE5128",  # vasc   rare (terracotta)
}

# ----------------------------------------------------------------------
# 1. Build allocation matrices live from the partitioners (numpy only)
# ----------------------------------------------------------------------
trainy = np.asarray(np.load(repo_root() / "dermamnist_64.npz")["train_labels"]).reshape(-1).astype(np.int64)


def alloc_matrix(partitioner, seed=42):
    clients, _ = partitioner(trainy, seed=seed)
    M = np.zeros((2, NUM_CLASSES), dtype=int)
    for cid, idxs in enumerate(clients):
        for c in range(NUM_CLASSES):
            M[cid, c] = int(np.sum(trainy[idxs] == c))
    return M


def js_divergence(p, q, base=2):
    """Jensen-Shannon divergence in [0, 1] for base 2 (no scipy dependency)."""
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / np.log(base)


M_l1 = alloc_matrix(two_client_86_14_quantity_only_stratified)
M_l4 = alloc_matrix(two_client_90_10_rare_stress)
JS_l1 = js_divergence(M_l1[0], M_l1[1])
JS_l4 = js_divergence(M_l4[0], M_l4[1])

# ----------------------------------------------------------------------
# 2. Plot
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 6.1))
fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.30, wspace=0.20)


def draw_panel(ax, M, main, sub, js, js_note, mark_rare_client=False):
    totals = M.sum(axis=1)
    pct = totals / totals.sum() * 100.0
    P = M / totals[:, None]
    x = np.arange(2)
    bottom = np.zeros(2)
    for c in range(NUM_CLASSES):
        ax.bar(x, P[:, c], bottom=bottom, color=COLOURS[c], edgecolor="white",
               linewidth=0.8, width=0.6, label=CLASS_SHORT[c])
        for i in range(2):
            if P[i, c] > 0.03:
                r, g, b, _ = to_rgba(COLOURS[c])
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(x[i], bottom[i] + P[i, c] / 2, f"{M[i, c]}", ha="center",
                        va="center", fontsize=8.2,
                        color=("white" if lum < 0.5 else INK))
        bottom += P[:, c]

    # frame the rare-only client (L4) so rare signal is tied to Client 1
    if mark_rare_client:
        ax.add_patch(Rectangle((1 - 0.33, 0.0), 0.66, 1.0, fill=False,
                               edgecolor="#8a3d1c", linewidth=1.8, zorder=6))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Client 0\n{pct[0]:.0f}%  (n = {totals[0]})",
                        f"Client 1\n{pct[1]:.0f}%  (n = {totals[1]})"], fontsize=9)
    if mark_rare_client:
        lbl = ax.get_xticklabels()[1]
        lbl.set_color("#8a3d1c"); lbl.set_fontweight("bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8.3)
    ax.set_ylabel("class proportion within client", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.18)

    # panel title + emphasis subtitle
    ax.text(0.5, 1.135, main, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color=INK)
    ax.text(0.5, 1.045, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.3, style="italic", color=MUTE)

    # prominent JS divergence (the quantitative L1/L4 contrast)
    ax.text(0.5, -0.27, f"JS divergence  {js}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.5, -0.355, js_note, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.4, style="italic", color=MUTE)


draw_panel(axA, M_l1, "L1 — quantity skew only",
           "near-global class mix on both clients", "≈ 0",
           "class mixtures near-identical → pure size skew")
draw_panel(axB, M_l4, "L4 — class-disjoint partition",
           "rare-class signal isolated on Client 1", "= 1",
           "class supports disjoint → label skew added", mark_rare_client=True)

# headline banner: the controlled variable shared by both panels
fig.text(0.5, 0.945,
         "Same 86 % / 14 % client-size skew in both panels — only the class structure differs",
         ha="center", va="center", fontsize=12, fontweight="bold", color=INK)

# grouped legend at the bottom, ordered common → rare → majority
order = [0, 1, 2, 3, 4, 6, 5]
handles, labels = axA.get_legend_handles_labels()
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
           ncol=7, fontsize=9, frameon=False, columnspacing=1.4, handlelength=1.1)
fig.text(0.5, 0.012,
         "grey = common classes      amber = rare-class set {df, mel, vasc}      slate = majority (mel-nevi)",
         ha="center", va="bottom", fontsize=8.6, color=MUTE)

OUT = thesis_figures_dir()
out_pdf = OUT / "F_l1_l4_partition_schematic.pdf"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
print(f"wrote {out_pdf}")
print(f"  L1 sizes {M_l1.sum(1).tolist()} JS={JS_l1:.2e}; L4 sizes {M_l4.sum(1).tolist()} JS={JS_l4:.4f}")
print(f"  L4 C0 classes {[CLASS_SHORT[c] for c in range(7) if M_l4[0,c]>0]}; "
      f"C1 classes {[CLASS_SHORT[c] for c in range(7) if M_l4[1,c]>0]}")
plt.close(fig)
