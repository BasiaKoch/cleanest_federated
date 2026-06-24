"""
Generate F_cross_experiment_per_class.pdf -- master per-class heatmap of the
FedProx/FedNova-minus-FedAvg macro-F1 difference across the principal
experiments.

Message: large positive differences concentrate in the rare classes, and only
under system heterogeneity; the majority class (mel-nevi) stays near zero.

Rows  : experiments, grouped into four labelled blocks (statistical
        heterogeneity / heterogeneity-ladder pilot / system heterogeneity /
        LR asymmetry).
Cols  : the seven DermaMNIST classes -- the rare-class set {dermato, melanoma,
        vascular} first (amber), then the common classes, then the majority
        class, then a macro-F1 summary column.
Cells : mean per-class (FedProx or FedNova) - FedAvg F1 difference.
        Numbers are printed only for large effects (|delta| >= 0.10); smaller
        cells rely on the colour scale.

Reads source CSVs directly from results/*/analysis/ and data/ (data preserved).
"""

from pathlib import Path
from fl_dermamnist.common.paths import package_root, thesis_figures_dir  # noqa: E402
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = package_root()
RESULTS = ROOT / "results"
DATA = ROOT / "results" / "thesis_ready" / "data"
OUT = thesis_figures_dir()

# CSV per-class column order == class-id order (akiec, bcc, bkl, df, mel, nv, vasc)
CLASS_COLS = ["f1_actinic", "f1_basal", "f1_benign_kerat", "f1_dermato",
              "f1_melanoma", "f1_mel_nevi", "f1_vascular"]
# Display order: rare-class set first, then common, then majority
ORDER = ["f1_dermato", "f1_melanoma", "f1_vascular",     # rare-class set (amber)
         "f1_actinic", "f1_basal", "f1_benign_kerat",    # common
         "f1_mel_nevi"]                                  # majority
COL_LABELS = ["dermato", "melanoma", "vascular", "actinic", "basal",
              "b-kerat", "mel-nevi"]
N_RARE = 3                       # first three columns are the rare-class set
NV_COL = 6                       # majority column

AMBER, AMBER_DK = "#E67E22", "#B9521E"
SLATE = "#2C3E50"
INK = "#1a1a1a"


def per_class_delta(df, algo_a, algo_b, where=None):
    if where is not None:
        df = df.query(where)
    a = df[df["algorithm"] == algo_a][CLASS_COLS].mean()
    b = df[df["algorithm"] == algo_b][CLASS_COLS].mean()
    return (b - a).values, (b.mean() - a.mean())


rows = []  # (group, label, delta_array[class-id order], macro_delta)

# --- statistical heterogeneity (10 seeds) ---
df = pd.read_csv(RESULTS / "iid" / "analysis" / "per_class_diff.csv")
d = df.sort_values("class_id")["mean_diff"].values
rows.append(("statistical", "IID baseline", d, d.mean()))
df = pd.read_csv(DATA / "per_class_delta.csv")
d = df.sort_values("class_id")["mean_delta"].values
rows.append(("statistical", "Engineered 90/10", d, d.mean()))
df = pd.read_csv(RESULTS / "dirichlet_a01" / "analysis" / "per_class_diff.csv")
d = df.sort_values("class_id")["mean_diff"].values
rows.append(("statistical", "Dirichlet α=0.1", d, d.mean()))

# --- heterogeneity ladder (1-seed pilot) ---
df = pd.read_csv(RESULTS / "heterogeneity_ladder" / "analysis" / "ladder_summary.csv")
for level, label in [("L0", "L0 (IID)"), ("L1", "L1 (quantity)"),
                     ("L2", "L2 (label-skew)"), ("L3", "L3 (mixed)"),
                     ("L4", "L4 (severe)")]:
    d, mac = per_class_delta(df[df["level"] == level], "fedavg", "fedprox")
    rows.append(("ladder", label, d, mac))

# --- system heterogeneity (3 seeds) ---
df = pd.read_csv(RESULTS / "node_pinned_L4" / "analysis" / "node_pinned_L4_summary.csv")
d, mac = per_class_delta(df, "fedavg", "fedprox")
rows.append(("system", "Node-pinned L4", d, mac))

df = pd.read_csv(RESULTS / "li2020_asymmetric_L4" / "analysis" / "li2020_asymmetric_L4_summary.csv")
fa = df[df["condition"].str.contains("FedAvg \\+ drop", regex=True)][CLASS_COLS].mean()
fp = df[df["condition"].str.contains("FedProx \\+ γ-inexact", regex=True)][CLASS_COLS].mean()
rows.append(("system", "Four-condition L4", (fp - fa).values, fp.mean() - fa.mean()))

df = pd.read_csv(RESULTS / "fedprox_perfect_storm_L4" / "analysis" / "perfect_storm_L4_summary.csv")
fa = df[df["condition"].str.contains("FedAvg \\+ drop", regex=True)][CLASS_COLS].mean()
fp = df[df["condition"].str.contains("μ=0.01", regex=True)][CLASS_COLS].mean()
rows.append(("system", "Perfect-storm L4", (fp - fa).values, fp.mean() - fa.mean()))

# --- LR asymmetry (3 seeds) ---
df = pd.read_csv(DATA / "asymmetric_lr_L4_summary.csv")
d1 = df[df["ratio_label"].str.contains("5:1")]
fa = d1[d1["algorithm"] == "fedavg"][CLASS_COLS].mean()
fp = d1[d1["algorithm"] == "fedprox"][CLASS_COLS].mean()
fn = d1[d1["algorithm"] == "fednova"][CLASS_COLS].mean()
rows.append(("lr", "5:1 LR · FedProx", (fp - fa).values, fp.mean() - fa.mean()))
rows.append(("lr", "5:1 LR · FedNova", (fn - fa).values, fn.mean() - fa.mean()))

# ----------------------------------------------------------------------
# Build matrix (rows × 8): 7 reordered classes + macro
# ----------------------------------------------------------------------
order_idx = [CLASS_COLS.index(c) for c in ORDER]
n_rows = len(rows)
M = np.zeros((n_rows, 8))
labels, groups = [], []
for i, (g, lab, d, mac) in enumerate(rows):
    M[i, :7] = np.asarray(d)[order_idx]
    M[i, 7] = mac
    labels.append(lab); groups.append(g)

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(12.4, 6.5))

VMAX = 0.30                                  # symmetric, centred at 0
cmap = plt.get_cmap("RdBu_r")                # colourblind-safe diverging
norm = mcolors.Normalize(vmin=-VMAX, vmax=VMAX)
ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)

# thin white gridlines between cells
ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.1)
ax.tick_params(which="minor", length=0)

# numbers only for large effects (|delta| >= 0.10); else rely on colour
for i in range(n_rows):
    for j in range(8):
        v = M[i, j]
        if abs(v) < 0.10:
            continue
        r, g, b, _ = cmap(norm(v))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                fontweight="bold", color=("white" if lum < 0.5 else INK))

# column labels (rare set amber, majority slate) + macro
ax.set_xticks(range(8))
ax.set_xticklabels(COL_LABELS + ["macro-F1"], fontsize=9)
for k, lab in enumerate(ax.get_xticklabels()):
    if k < N_RARE:
        lab.set_color(AMBER_DK); lab.set_fontweight("bold")
    elif k == NV_COL:
        lab.set_color(SLATE)
    elif k == 7:
        lab.set_fontweight("bold")

# column-group headers above the heatmap
ax.set_ylim(n_rows - 0.5, -1.7)
ax.text(1.0, -1.15, "rare-class set", ha="center", va="center", fontsize=9,
        fontweight="bold", color=AMBER_DK)
ax.text(4.0, -1.15, "common", ha="center", va="center", fontsize=9, color="0.35")
ax.text(6.0, -1.15, "majority", ha="center", va="center", fontsize=9, color=SLATE)
for x0, x1, c in [(-0.4, 2.4, AMBER), (2.6, 5.4, "0.6"), (5.6, 6.4, SLATE)]:
    ax.plot([x0, x1], [-0.62, -0.62], color=c, lw=1.6, clip_on=False)

# vertical separators (group boundaries; macro boundary heavier)
for x in (2.5, 5.5):
    ax.axvline(x, color="0.45", linewidth=1.0)
ax.axvline(6.5, color="black", linewidth=1.5)

# row labels + horizontal group separators
ax.set_yticks(range(n_rows))
ax.set_yticklabels(labels, fontsize=9)
for i in range(1, n_rows):
    if groups[i] != groups[i - 1]:
        ax.axhline(i - 0.5, color="black", linewidth=1.5)

# group block labels on the right
group_names = {"statistical": "Statistical\nheterogeneity\n(10 seeds)",
               "ladder": "Heterogeneity\nladder\n(1-seed pilot)",
               "system": "System\nheterogeneity\n(3 seeds)",
               "lr": "LR asymmetry\n(3 seeds)"}
seen = set()
for i, g in enumerate(groups):
    if g in seen:
        continue
    seen.add(g)
    span = groups.count(g)
    ax.text(7.75, i + (span - 1) / 2, group_names[g], ha="left", va="center",
            fontsize=8.5, color="0.25", style="italic")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.20,
                    label="per-class F1 difference  (method $-$ FedAvg)")
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([-0.3, -0.15, 0, 0.15, 0.3])
cbar.set_ticklabels(["≤ −0.30", "−0.15", "0", "+0.15", "≥ +0.30"])

ax.set_title("Gains over FedAvg concentrate in the rare classes — only under system heterogeneity",
             loc="left", fontweight="bold", fontsize=12, pad=26)

fig.tight_layout(rect=[0, 0.0, 1.0, 0.98])
for ext in ("pdf",):
    out_path = OUT / f"F_cross_experiment_per_class.{ext}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"wrote {out_path}")
plt.close(fig)
