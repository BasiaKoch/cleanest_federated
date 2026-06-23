"""
Generate F_heterogeneity_escalation.pdf — a single figure showing the
FedAvg vs FedProx gap growing as heterogeneity / system stress increases.

Source data (all multi-seed; mean ± SD across 3 or 10 seeds):
  IID baseline                 results/thesis_ready/data/per_seed_results.csv  (10 seeds, but separate from this; use IID row from partition-robustness)
  L4 symmetric (node-pinned)   results/node_pinned_L4/analysis/node_pinned_L4_summary.csv
  Li 2020 §5.2 asymmetric      results/li2020_asymmetric_L4/analysis/li2020_asymmetric_L4_summary.csv
  Perfect-storm μ=1.0          results/fedprox_perfect_storm_L4/analysis/perfect_storm_L4_summary.csv
  Perfect-storm μ=0.01         (same source)

Numbers below are computed once from the CSVs and reproduced here verbatim
to keep the figure self-contained.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Verified means and SDs from the source CSVs
# ----------------------------------------------------------------------
conditions = [
    # label_short (multi-line), FedAvg mean, FedAvg sd, FedProx mean, FedProx sd, n_seeds, fp_variant
    ("IID baseline\n(no heterogeneity)\n$\\mu = 0.01$",                       0.585, 0.020, 0.579, 0.019, 10, ""),
    ("L4 symmetric\n(stat. het. only)\n$\\mu = 0.01$",                        0.492, 0.019, 0.486, 0.019,  3, ""),
    ("Four-condition L4\n(+ straggler asym.)\nFP: $\\gamma$-inexact",          0.364, 0.009, 0.479, 0.010,  3, ""),
    ("Perfect-storm L4\n(90% stragglers)\nFP: $\\mu{=}1.0$, $\\gamma$-inex.", 0.087, 0.049, 0.365, 0.023, 3, ""),
    ("Perfect-storm L4\n(90% stragglers)\nFP: $\\mu{=}0.01$, $\\gamma$-inex.", 0.087, 0.049, 0.491, 0.003, 3, ""),
]

labels       = [c[0] for c in conditions]
fa_means     = np.array([c[1] for c in conditions])
fa_sds       = np.array([c[2] for c in conditions])
fp_means     = np.array([c[3] for c in conditions])
fp_sds       = np.array([c[4] for c in conditions])
deltas       = fp_means - fa_means

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 5.0),
                                gridspec_kw=dict(width_ratios=[1.75, 1.0],
                                                  wspace=0.28))

# ------------------ Panel A: grouped bars ------------------
x       = np.arange(len(labels))
width   = 0.36
col_fa  = "#3b6e8f"
col_fp  = "#c97644"

bars_fa = axA.bar(x - width/2, fa_means, width,
                  yerr=fa_sds, label="FedAvg",
                  color=col_fa, alpha=0.92, capsize=3,
                  error_kw=dict(ecolor="0.2", elinewidth=0.8, capthick=0.8))
bars_fp = axA.bar(x + width/2, fp_means, width,
                  yerr=fp_sds, label="FedProx",
                  color=col_fp, alpha=0.92, capsize=3,
                  error_kw=dict(ecolor="0.2", elinewidth=0.8, capthick=0.8))

# Annotate each bar pair with its delta
for xi, fa, fp, d in zip(x, fa_means, fp_means, deltas):
    y_anchor = max(fa, fp) + 0.05
    color    = "#1e7d3a" if d > 0.05 else ("#888888" if abs(d) < 0.02 else "#1e7d3a")
    sign     = "+" if d >= 0 else ""
    axA.annotate(f"$\\Delta = {sign}{d:.3f}$",
                 xy=(xi, y_anchor), ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold",
                 color=color)

axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=8.5)
axA.set_ylabel("Test macro-F1 (mean $\\pm$ sample SD)")
axA.set_ylim(0, 0.78)
axA.set_yticks(np.arange(0, 0.81, 0.1))
axA.axhline(0, color="0.3", linewidth=0.6)
axA.grid(axis="y", linestyle=":", alpha=0.4)
axA.legend(loc="upper right", framealpha=0.95, fontsize=10)
axA.set_title("(A) Macro-F1 of FedAvg vs FedProx as system stress escalates",
              loc="left", fontweight="bold", fontsize=11, pad=8)

# Add a subtle horizontal arrow indicating direction of increasing stress
axA.annotate("", xy=(len(labels) - 0.6, 0.74), xytext=(0.0, 0.74),
             arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))
axA.text((len(labels) - 1) / 2, 0.755,
         "increasing heterogeneity / system stress",
         ha="center", va="bottom", fontsize=9, style="italic", color="0.4")

# ------------------ Panel B: within-pair Δ ------------------
short_labels = [
    "IID",
    "L4 sym.",
    "Four-cond. L4",
    "P.-storm $\\mu{=}1.0$",
    "P.-storm $\\mu{=}0.01$",
]
xB     = np.arange(len(deltas))
colors = ["#888888" if abs(d) < 0.02 else "#1e7d3a" for d in deltas]
barsB  = axB.barh(xB, deltas, color=colors, alpha=0.92,
                  edgecolor="0.25", linewidth=0.4)

# Always place the numeric label to the RIGHT of the bar's end
# (for small negative bars this means the label sits just past 0 on the positive side)
for yi, d in zip(xB, deltas):
    sign     = "+" if d > 0 else ("−" if d < 0 else "")
    text_x   = max(d, 0) + 0.010
    color    = "#888888" if abs(d) < 0.02 else "#155a2a"
    weight   = "normal" if abs(d) < 0.02 else "bold"
    axB.text(text_x, yi, f"{sign}{abs(d):.3f}",
             ha="left", va="center", fontsize=10, fontweight=weight,
             color=color)

axB.set_yticks(xB)
axB.set_yticklabels(short_labels, fontsize=9.5)
axB.invert_yaxis()
axB.axvline(0, color="0.25", linewidth=0.8)
axB.set_xlabel("FedProx $-$ FedAvg macro-F1")
axB.set_xlim(-0.025, 0.48)
axB.grid(axis="x", linestyle=":", alpha=0.4)
axB.set_title("(B) Within-pair $\\Delta$ across stress conditions",
              loc="left", fontweight="bold", fontsize=11, pad=8)

# ------------------ Caption / write ------------------
fig.tight_layout(rect=[0, 0.02, 1, 0.97])

for ext in ("pdf", "png"):
    out_path = OUT / f"F_heterogeneity_escalation.{ext}"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"wrote {out_path}")

plt.close(fig)
