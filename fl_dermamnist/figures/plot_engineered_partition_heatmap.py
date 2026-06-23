"""Generate F_engineered_partition_heatmap.pdf -- class-by-client heatmap
for the engineered balanced-paired 7-client partition (Methods).

Builds the heatmap directly from the partition spec in
fl_dermamnist/data/partition.py so the figure is repo-grounded (and, when the
dataset npz is present at the repo root, cross-checks against the live
partition).

Layout:
  rows    = 7 DermaMNIST classes (full name + abbreviation)
  columns = Client 0 .. Client 6
  cells   = training-sample counts (log colour so minority cells stay visible
            despite mel-nevi's ~670-per-client mass; every count is annotated)
  bottom  = per-client total training samples, column-aligned with the heatmap
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Make the package importable when run directly, regardless of this file's
# depth (walk up to the directory that contains the fl_dermamnist package).
_HERE = Path(__file__).resolve()
for _cand in _HERE.parents:
    if (_cand / "fl_dermamnist" / "__init__.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from fl_dermamnist.common.paths import repo_root, thesis_figures_dir
from fl_dermamnist.data.partition import BALANCED_PAIRED_7_CLIENTS_SPEC, balanced_paired_7_clients


CLASS_LONG = [
    "actinic keratoses (akiec)",
    "basal cell carcinoma (bcc)",
    "benign keratosis (bkl)",
    "dermatofibroma (df)",
    "melanoma (mel)",
    "mel-nevi (nv, majority)",
    "vascular lesions (vasc)",
]
RARE_ROWS = (3, 4, 6)            # df, mel, vasc -- the rare-class group
NUM_CLASSES = 7
NUM_CLIENTS = 7

# thesis palette: teal sequential heatmap + amber rare-class accent
AMBER, AMBER_DK = "#E67E22", "#B9521E"
INK = "#1b1b1b"
TEAL = LinearSegmentedColormap.from_list(
    "thesis_teal", ["#cfe8e0", "#8ecabd", "#46a392", "#11806a", "#0a4a3f"])
TEAL.set_bad("#f4f4f4")          # faint grey for empty cells (distinct from low teal)

# ----------------------------------------------------------------------
# 1. Build the allocation matrix straight from the spec
# ----------------------------------------------------------------------
M = np.zeros((NUM_CLASSES, NUM_CLIENTS), dtype=int)
for entry in BALANCED_PAIRED_7_CLIENTS_SPEC:
    cid = entry["id"]
    for c, n in entry["per_class"].items():
        M[c, cid] = n
per_client_total = M.sum(axis=0)

# Optional cross-check against the live partition (needs torch + the dataset
# npz); the spec is the source of truth, so a missing dependency just skips it.
ROOT_NPZ = repo_root() / "dermamnist_64.npz"
if ROOT_NPZ.exists():
    try:
        from fl_dermamnist.data.load import load_dermmnist
        train, _, _ = load_dermmnist(str(ROOT_NPZ))
        clients_idx, _ = balanced_paired_7_clients(train.labels, seed=42)
        assert per_client_total.tolist() == [len(s) for s in clients_idx], \
            "spec/live partition size mismatch"
    except ModuleNotFoundError:
        pass

# ----------------------------------------------------------------------
# 2. Plot
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                     "pdf.fonttype": 42})

fig = plt.figure(figsize=(7.4, 5.1))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.035], height_ratios=[7, 1.35],
                      hspace=0.09, wspace=0.03)
ax = fig.add_subplot(gs[0, 0])                 # heatmap
cax = fig.add_subplot(gs[0, 1])                # colorbar (heatmap height only)
axb = fig.add_subplot(gs[1, 0], sharex=ax)     # client-size bar (column-aligned)

vmin, vmax = int(M[M > 0].min()), int(M.max())
norm = LogNorm(vmin=vmin, vmax=vmax)
im = ax.imshow(np.ma.masked_equal(M, 0), aspect="auto", cmap=TEAL, norm=norm)

# annotate every non-zero cell; text colour by cell luminance
for c in range(NUM_CLASSES):
    for cid in range(NUM_CLIENTS):
        v = M[c, cid]
        if v == 0:
            continue
        r, g, b, _ = TEAL(norm(v))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(cid, c, f"{v}", ha="center", va="center", fontsize=8.4,
                color=("white" if lum < 0.55 else INK))

# thin white gridlines between cells
ax.set_xticks(np.arange(-0.5, NUM_CLIENTS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.3)
ax.tick_params(which="minor", length=0)

ax.set_yticks(range(NUM_CLASSES))
ax.set_yticklabels(CLASS_LONG, fontsize=10)
for i, lab in enumerate(ax.get_yticklabels()):
    if i in RARE_ROWS:
        lab.set_color(AMBER_DK); lab.set_fontweight("bold")
ax.tick_params(axis="x", which="major", bottom=False, labelbottom=False, top=False)
ax.tick_params(axis="y", which="major", left=False)
for sp in ax.spines.values():
    sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.6)

# elegant amber outline around each rare-class row
for r in RARE_ROWS:
    ax.add_patch(plt.Rectangle((-0.5, r - 0.5), NUM_CLIENTS, 1, fill=False,
                               edgecolor=AMBER, linewidth=1.7, zorder=6))

# concise structural annotation in the empty upper-right block
note = ("Each minority class → exactly two clients\n"
        "    df: C2·C3      mel & vasc: C4·C5\n"
        "mel-nevi (majority) → shared by all seven")
ax.text(6.43, -0.40, note, ha="right", va="top", fontsize=8.3, color=INK,
        linespacing=1.5, zorder=7,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=AMBER, lw=0.9, alpha=0.96))

# colorbar with plain, meaningful ticks
cbar = fig.colorbar(im, cax=cax)
ticks = [t for t in (40, 100, 200, 400, 670) if vmin <= t <= vmax]
cbar.set_ticks(ticks)
cbar.set_ticklabels([str(t) for t in ticks])
cbar.ax.tick_params(labelsize=8, length=2)
cbar.set_label("samples per cell  (log colour scale)", fontsize=8.5)
cbar.outline.set_linewidth(0.5)

# bottom: per-client totals, aligned with the heatmap columns
axb.bar(range(NUM_CLIENTS), per_client_total, color="#46a392",
        edgecolor="white", linewidth=0.6, width=0.86)
for cid, tot in enumerate(per_client_total):
    axb.text(cid, tot + vmax * 0.06, f"{tot}", ha="center", va="bottom",
             fontsize=7.8, color="#555555")
axb.set_xticks(range(NUM_CLIENTS))
axb.set_xticklabels([f"C{i}" for i in range(NUM_CLIENTS)], fontsize=9.5)
axb.set_ylim(0, max(per_client_total) * 1.24)
axb.set_ylabel("client size\n(samples)", fontsize=8.8)
axb.set_yticks([0, 1000])
axb.tick_params(axis="y", labelsize=7.6)
axb.tick_params(axis="x", top=False, length=2)
axb.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)
for side in ("top", "right"):
    axb.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axb.spines[side].set_linewidth(0.6)

ax.set_title("Engineered balanced-paired 7-client partition: class × client allocation",
             fontsize=11, pad=9, color=INK)

OUT = thesis_figures_dir()
out_pdf = OUT / "F_engineered_partition_heatmap.pdf"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out_pdf}")
plt.close(fig)
