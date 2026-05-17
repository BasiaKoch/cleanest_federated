"""Best-vs-last round bar chart, comparable to Marija (2025) Fig 3.7 right panel.

Two grouped bars per algorithm: peak val_macro_F1 vs final round val_macro_F1.
Shows that FedProx maintains a higher plateau but post-peak drop is similar
in absolute magnitude.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT     = Path(__file__).resolve().parent.parent
FIG_DIR  = ROOT / "figures"
DATA_DIR = ROOT / "data"

d = json.load(open(DATA_DIR / "best_vs_last_round.json"))
fa = d["summary"]["fedavg"]
fp = d["summary"]["fedprox"]

C_FA = "#2b6cb0"
C_FP = "#dd6b20"

plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

labels = ["FedAvg", "FedProx"]
peak_means  = [fa["mean_peak_val_f1"],  fp["mean_peak_val_f1"]]
peak_sds    = [fa["sd_peak_val_f1"],    fp["sd_peak_val_f1"]]
final_means = [fa["mean_final_val_f1"], fp["mean_final_val_f1"]]
final_sds   = [fa["sd_final_val_f1"],   fp["sd_final_val_f1"]]

x = np.arange(2)
w = 0.36
b1 = ax.bar(x - w/2, peak_means,  w, yerr=peak_sds,  capsize=6,
            color=[C_FA, C_FP], edgecolor="black",
            label="Peak round (best val)",
            hatch="//", alpha=0.95)
b2 = ax.bar(x + w/2, final_means, w, yerr=final_sds, capsize=6,
            color=[C_FA, C_FP], edgecolor="black",
            label="Final round (R=150)",
            alpha=0.55)

for bars, vals in [(b1, peak_means), (b2, final_means)]:
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}",
                    (bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

# Annotate the drop magnitudes
for i, (p, f) in enumerate(zip(peak_means, final_means)):
    ax.annotate(f"drop = -{p-f:.3f}",
                xy=(i, (p + f) / 2),
                ha="center", fontsize=9, color="#4a5568", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Validation macro-F1 (mean ± SD across 10 seeds)")
ax.set_ylim(0, max(peak_means) + max(peak_sds) + 0.08)
ax.set_title("Peak vs final round comparison\n"
             "FedProx maintains a higher plateau; post-peak drop is similar in magnitude")
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax.grid(alpha=0.25, axis="y")
ax.spines[["top", "right"]].set_visible(False)

out_png = FIG_DIR / "11_best_vs_last.png"
out_pdf = FIG_DIR / "11_best_vs_last.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf,           bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")
