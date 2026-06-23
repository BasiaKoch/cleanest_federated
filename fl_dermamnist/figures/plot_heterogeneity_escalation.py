"""
Generate F_heterogeneity_escalation.pdf -- synthesis figure: the FedProx-vs-FedAvg
macro-F1 gap is regime-driven, near-zero under weak stress and large only under
specific straggler regimes.

Source data (all multi-seed; mean +/- SD across 3 or 10 seeds):
  IID baseline                 results/thesis_ready/data/per_seed_results.csv  (10 seeds)
  L4 symmetric (node-pinned)   results/node_pinned_L4/analysis/node_pinned_L4_summary.csv
  Li 2020 5.2 asymmetric       results/li2020_asymmetric_L4/analysis/li2020_asymmetric_L4_summary.csv
  Perfect-storm mu=1.0/0.01    results/fedprox_perfect_storm_L4/analysis/perfect_storm_L4_summary.csv

Numbers below are computed once from the CSVs and reproduced here verbatim
to keep the figure self-contained.
"""

from pathlib import Path
from fl_dermamnist.common.paths import thesis_figures_dir  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt

OUT = thesis_figures_dir()
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Verified means and SDs from the source CSVs (unchanged)
#   short_label, FedAvg mean, FedAvg sd, FedProx mean, FedProx sd, n_seeds
# ----------------------------------------------------------------------
conditions = [
    ("IID",                       0.585, 0.020, 0.579, 0.019, 10),
    ("L4",                        0.492, 0.019, 0.486, 0.019,  3),
    ("L4\n+ straggler",           0.364, 0.009, 0.479, 0.010,  3),
    ("Perfect-storm\n$\\mu=1.0$", 0.087, 0.049, 0.365, 0.023,  3),
    ("Perfect-storm\n$\\mu=0.01$",0.087, 0.049, 0.491, 0.003,  3),
]
B_labels = ["IID", "L4", "L4 + straggler",
            "Perfect-storm $\\mu{=}1.0$", "Perfect-storm $\\mu{=}0.01$"]

labels   = [c[0] for c in conditions]
fa_means = np.array([c[1] for c in conditions])
fa_sds   = np.array([c[2] for c in conditions])
fp_means = np.array([c[3] for c in conditions])
fp_sds   = np.array([c[4] for c in conditions])
deltas   = fp_means - fa_means
NOISE    = 0.02                       # |Δ| below this is within seed noise
is_eff   = np.abs(deltas) >= NOISE    # [F, F, T, T, T]
SPLIT    = int(np.argmax(is_eff))     # first effect cell -> boundary at SPLIT-0.5

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------
COL_FA, COL_FP = "#3b6e8f", "#c97644"     # FedAvg / FedProx
GREEN, GREY    = "#1e7d3a", "#8a8a8a"     # large effect / within noise
NOISE_BG, EFF_BG = "#f0f0f0", "#eaf4ee"   # zone backgrounds
DIV = "#c4c4c4"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                               gridspec_kw=dict(width_ratios=[1.75, 1.0], wspace=0.26))

# ====================== Panel A: grouped bars =========================
x, width = np.arange(len(labels)), 0.38
b = SPLIT - 0.5

# zone backgrounds + divider (drawn first, behind everything)
axA.axvspan(-0.6, b, color=NOISE_BG, zorder=0)
axA.axvspan(b, len(labels) - 0.4, color=EFF_BG, zorder=0)
axA.axvline(b, color=DIV, ls=(0, (4, 3)), lw=1.0, zorder=1)

ekw = dict(ecolor="0.25", elinewidth=0.9, capthick=0.9)
axA.bar(x - width/2, fa_means, width, yerr=fa_sds, label="FedAvg",
        color=COL_FA, capsize=3.5, error_kw=ekw, zorder=3)
axA.bar(x + width/2, fp_means, width, yerr=fp_sds, label="FedProx",
        color=COL_FP, capsize=3.5, error_kw=ekw, zorder=3)

# direct Δ label above each pair, coloured by noise/effect
for xi, fa, fas, fp, fps, d, eff in zip(x, fa_means, fa_sds, fp_means, fp_sds, deltas, is_eff):
    y = max(fa + fas, fp + fps) + 0.028
    axA.annotate(f"$\\Delta={'+' if d >= 0 else '-'}{abs(d):.3f}$",
                 xy=(xi, y), ha="center", va="bottom", fontsize=9.5,
                 fontweight=("bold" if eff else "normal"),
                 color=(GREEN if eff else GREY))

# zone labels
axA.text((SPLIT - 1) / 2, 0.70, "within seed noise", ha="center", va="center",
         fontsize=9, style="italic", color=GREY)
axA.text((SPLIT + len(labels) - 1) / 2, 0.70, "large FedProx advantage",
         ha="center", va="center", fontsize=9, style="italic", color=GREEN)

# clear "increasing stress" arrow across the top
axA.annotate("", xy=(len(labels) - 0.5, 0.80), xytext=(-0.5, 0.80),
             arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.6))
axA.text((len(labels) - 1) / 2, 0.815, "increasing heterogeneity / system stress",
         ha="center", va="bottom", fontsize=9.3, style="italic", color="0.4")

axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel("Test macro-F1  (mean $\\pm$ SD)")
axA.set_ylim(0, 0.88); axA.set_yticks(np.arange(0, 0.81, 0.2))
axA.set_xlim(-0.6, len(labels) - 0.4)
axA.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
axA.legend(loc="center", bbox_to_anchor=(0.74, 0.60), framealpha=0.96,
           edgecolor="0.8", fontsize=9.5, handlelength=1.3)
axA.set_title("(A) FedAvg vs FedProx macro-F1 as stress escalates",
              loc="left", fontweight="bold", fontsize=11, pad=8)

# ====================== Panel B: within-pair Δ ========================
xB = np.arange(len(deltas))
colorsB = [GREEN if e else GREY for e in is_eff]

# match Panel A: shade the noise rows vs effect rows + a seed-noise band at 0
axB.axhspan(-0.5, b, color=NOISE_BG, zorder=0)
axB.axhspan(b, len(deltas) - 0.5, color=EFF_BG, zorder=0)
axB.axvspan(-NOISE, NOISE, color="0.82", alpha=0.5, zorder=1)
axB.axvline(0, color="0.25", linewidth=0.9, zorder=2)

axB.barh(xB, deltas, color=colorsB, edgecolor="0.25", linewidth=0.4, zorder=3)
for yi, d, eff in zip(xB, deltas, is_eff):
    axB.text(max(d, 0) + 0.012, yi, f"{'+' if d > 0 else '−'}{abs(d):.3f}",
             ha="left", va="center", fontsize=9.5,
             fontweight=("bold" if eff else "normal"),
             color=(GREEN if eff else GREY))

axB.text(0.0, -0.8, "seed noise ($|\\Delta|{<}0.02$)", ha="center", va="center",
         fontsize=7.6, color="0.5")
axB.set_yticks(xB); axB.set_yticklabels(B_labels, fontsize=9)
axB.invert_yaxis()
axB.set_xlabel("FedProx $-$ FedAvg macro-F1")
axB.set_xlim(-0.03, 0.49); axB.set_ylim(len(deltas) - 0.5, -1.0)
axB.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
axB.set_title("(B) Within-pair $\\Delta$", loc="left", fontweight="bold",
              fontsize=11, pad=8)

fig.tight_layout(rect=[0, 0.01, 1, 0.98])
for ext in ("pdf", "png"):
    out_path = OUT / f"F_heterogeneity_escalation.{ext}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"wrote {out_path}")
plt.close(fig)
