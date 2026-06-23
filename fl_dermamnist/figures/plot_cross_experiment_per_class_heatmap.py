"""
Generate F_cross_experiment_per_class.pdf --- a master heatmap showing the
per-class FedProx-minus-FedAvg macro-F1 delta across the principal
experiments of the thesis.

Rows: experiments (~10), grouped by heterogeneity type
Cols: the seven DermaMNIST classes (ordered: rare → common)
Cells: mean (FedProx - FedAvg) per-class F1, colour-coded; macro-F1 Δ
appended on the right edge as a separate "macro" column.

Reads source CSVs directly from results/*/analysis/ and data/.
"""

from pathlib import Path
from fl_dermamnist.common.paths import repo_root, package_root, results_root, thesis_ready_root, thesis_data_dir, thesis_figures_dir  # noqa: E402
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = package_root()
RESULTS = ROOT / "results"
DATA = ROOT / "results" / "thesis_ready" / "data"
OUT = thesis_figures_dir()

CLASS_COLS = ["f1_actinic", "f1_basal", "f1_benign_kerat", "f1_dermato",
              "f1_melanoma", "f1_mel_nevi", "f1_vascular"]
CLASS_LABELS = ["actinic\n(n=66)", "basal\n(n=103)", "b-kerat\n(n=220)",
                "dermato\n(n=23)", "melanoma\n(n=223)", "mel-nevi\n(n=1341)",
                "vascular\n(n=29)"]
# Reorder: rare → common by test support
ORDER = ["f1_dermato", "f1_vascular", "f1_actinic", "f1_basal",
         "f1_b-kerat" if False else "f1_benign_kerat",
         "f1_melanoma", "f1_mel_nevi"]
ORDER_LABELS = ["dermato\nn=23", "vascular\nn=29", "actinic\nn=66",
                "basal\nn=103", "b-kerat\nn=220", "melanoma\nn=223",
                "mel-nevi\nn=1341"]


def per_class_delta_from_summary(df, algo_a, algo_b, where=None):
    """Compute mean per-class delta (algo_b - algo_a) from a long-form CSV."""
    if where is not None:
        df = df.query(where)
    a = df[df["algorithm"] == algo_a][CLASS_COLS].mean()
    b = df[df["algorithm"] == algo_b][CLASS_COLS].mean()
    return (b - a).values, (b.mean() - a.mean())  # macro = mean of per-class


rows = []  # list of (group, label, n_seed_str, delta_array, macro_delta)

# 1. IID baseline (from per_class_diff.csv)
df = pd.read_csv(RESULTS / "iid" / "analysis" / "per_class_diff.csv")
deltas = df.sort_values("class_id")["mean_diff"].values
rows.append(("statistical", "IID baseline", "10 s",
             deltas, deltas.mean()))

# 2. Headline engineered
df = pd.read_csv(DATA / "per_class_delta.csv")
deltas = df.sort_values("class_id")["mean_delta"].values
rows.append(("statistical", "Engineered 90/10", "10 s",
             deltas, deltas.mean()))

# 3. Dirichlet α=0.1
df = pd.read_csv(RESULTS / "dirichlet_a01" / "analysis" / "per_class_diff.csv")
deltas = df.sort_values("class_id")["mean_diff"].values
rows.append(("statistical", "Dirichlet α=0.1", "10 s",
             deltas, deltas.mean()))

# 4. Heterogeneity ladder (5 levels) — per-class from ladder_summary
df = pd.read_csv(RESULTS / "heterogeneity_ladder" / "analysis" / "ladder_summary.csv")
ladder_levels = [("L0", "L0 (IID 50/50)"),
                 ("L1", "L1 (Quantity 86/14)"),
                 ("L2", "L2 (Label-skew)"),
                 ("L3", "L3 (Mixed 70/30)"),
                 ("L4", "L4 (Severe 90/10)")]
for level, label in ladder_levels:
    sub = df[df["level"] == level]
    deltas, mac = per_class_delta_from_summary(sub, "fedavg", "fedprox")
    rows.append(("ladder", label, "1 s pilot", deltas, mac))

# 5. Node-pinned L4 (3 seeds)
df = pd.read_csv(RESULTS / "node_pinned_L4" / "analysis" / "node_pinned_L4_summary.csv")
deltas, mac = per_class_delta_from_summary(df, "fedavg", "fedprox")
rows.append(("system", "Node-pinned L4 (sym.)", "3 s", deltas, mac))

# 6. Li 2020 §5.2 — FedAvg+drop vs FedProx+γ-inexact (3 seeds)
df = pd.read_csv(RESULTS / "li2020_asymmetric_L4" / "analysis" / "li2020_asymmetric_L4_summary.csv")
fa = df[df["condition"].str.contains("FedAvg \\+ drop", regex=True)][CLASS_COLS].mean()
fp = df[df["condition"].str.contains("FedProx \\+ γ-inexact", regex=True)][CLASS_COLS].mean()
deltas = (fp - fa).values
mac = (fp.mean() - fa.mean())
rows.append(("system", "Four-condition L4", "3 s", deltas, mac))

# 7. Perfect-storm L4 — FedAvg+drop vs FedProx μ=0.01 + γ-inexact (3 seeds)
df = pd.read_csv(RESULTS / "fedprox_perfect_storm_L4" / "analysis" / "perfect_storm_L4_summary.csv")
fa = df[df["condition"].str.contains("FedAvg \\+ drop", regex=True)][CLASS_COLS].mean()
fp = df[df["condition"].str.contains("μ=0.01", regex=True)][CLASS_COLS].mean()
deltas = (fp - fa).values
mac = (fp.mean() - fa.mean())
rows.append(("system", "Perfect-storm L4", "3 s", deltas, mac))

# 8. D1 LR asymmetry at 5:1 (FedProx vs FedAvg)
df = pd.read_csv(DATA / "asymmetric_lr_L4_summary.csv")
d1_51 = df[df["ratio_label"].str.contains("5:1")]
fa = d1_51[d1_51["algorithm"] == "fedavg"][CLASS_COLS].mean()
fp = d1_51[d1_51["algorithm"] == "fedprox"][CLASS_COLS].mean()
deltas = (fp - fa).values
mac = (fp.mean() - fa.mean())
rows.append(("system", "D1 5:1 LR (FP vs FA)", "3 s", deltas, mac))

# 9. D1 LR asymmetry at 5:1 (FedNova vs FedAvg) — extra row to highlight FedNova rescue
fn = d1_51[d1_51["algorithm"] == "fednova"][CLASS_COLS].mean()
deltas_fn = (fn - fa).values
mac_fn = (fn.mean() - fa.mean())
rows.append(("system", "D1 5:1 LR (FN vs FA)", "3 s", deltas_fn, mac_fn))

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
})

n_rows = len(rows)
fig, ax = plt.subplots(figsize=(11.5, 6.0))

# Reorder columns: rare → common
order_idx = [CLASS_COLS.index(c) for c in ORDER]

# Build the matrix: n_rows × 8 (7 classes + macro)
M = np.zeros((n_rows, 8))
labels = []
groups = []
n_seeds = []
for i, (g, lab, ns, deltas, mac) in enumerate(rows):
    M[i, :7] = deltas[order_idx]
    M[i, 7] = mac
    labels.append(lab)
    groups.append(g)
    n_seeds.append(ns)

# Symmetric diverging colormap; clip at ±0.3 for readability
vmax = 0.30
cmap = plt.get_cmap("RdBu_r")
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)

# Annotate cells with the delta value
for i in range(n_rows):
    for j in range(8):
        v = M[i, j]
        if abs(v) < 0.005:
            txt = "≈0"
        else:
            txt = f"{v:+.2f}" if abs(v) >= 0.10 else f"{v:+.3f}"
        # Choose colour for legibility
        rgba = cmap(norm(v))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_col = "white" if lum < 0.45 else "black"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=7.5, color=text_col, fontweight="bold" if abs(v) >= 0.10 else "normal")

# Tick labels
ax.set_xticks(np.arange(8))
ax.set_xticklabels(ORDER_LABELS + ["macro-F1\n(mean)"], fontsize=8.5)
# Highlight macro column boundary
ax.axvline(6.5, color="black", linewidth=1.4)

# Row labels: experiment name + seed count in parens
yt_labels = [f"{lab}  ({ns})" for lab, ns in zip(labels, n_seeds)]
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(yt_labels, fontsize=9)

# Horizontal separators between groups
prev_grp = groups[0]
for i in range(1, n_rows):
    if groups[i] != prev_grp:
        ax.axhline(i - 0.5, color="black", linewidth=1.0)
        prev_grp = groups[i]

# Group brackets on the right (text annotations)
group_names = {"statistical": "Statistical\nheterogeneity",
               "ladder":      "Heterogeneity\nladder (1 seed)",
               "system":      "System\nheterogeneity"}
seen = set()
for i, g in enumerate(groups):
    if g in seen:
        continue
    seen.add(g)
    span = sum(1 for x in groups if x == g)
    ax.text(8.05, i + (span - 1) / 2, group_names[g],
            ha="left", va="center", fontsize=8.5,
            color="0.30", style="italic")

# Colourbar
cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.16,
                    label="FedProx (or FedNova) $-$ FedAvg, per-class F1")
cbar.ax.tick_params(labelsize=8)

ax.set_title("Per-class FedProx-vs-FedAvg differences across the principal experiments",
             loc="left", fontweight="bold", fontsize=11.5, pad=10)

# Caption-style note inside the figure
ax.set_xlabel("Class (ordered by test-set support, rare $\\rightarrow$ common)",
              fontsize=10, labelpad=6)

fig.tight_layout(rect=[0, 0.02, 1.00, 0.98])

for ext in ("pdf",):
    out_path = OUT / f"F_cross_experiment_per_class.{ext}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"wrote {out_path}")

plt.close(fig)
