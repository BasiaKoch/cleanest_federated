"""Per-client specialty bar chart from per_client_specialty.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT     = Path(__file__).resolve().parent.parent
FIG_DIR  = ROOT / "figures"
DATA_DIR = ROOT / "data"

df = pd.read_csv(DATA_DIR / "per_client_specialty.csv")
# Don't include "all minorities" (it overlaps with the specialty pairs)
df = df[df.specialty != "all minorities"].reset_index(drop=True)

C_FA = "#2b6cb0"
C_FP = "#dd6b20"
C_GAIN = "#38a169"

plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1.2, 1]})

# Left panel: grouped bars FedAvg vs FedProx per specialty
x = np.arange(len(df))
w = 0.36
b1 = ax1.bar(x - w/2, df["fedavg_mean_f1"],  w, yerr=df["fedavg_sd_f1"],
             capsize=5, color=C_FA, edgecolor="black", label="FedAvg")
b2 = ax1.bar(x + w/2, df["fedprox_mean_f1"], w, yerr=df["fedprox_sd_f1"],
             capsize=5, color=C_FP, edgecolor="black", label="FedProx")

for bars, vals in [(b1, df["fedavg_mean_f1"]), (b2, df["fedprox_mean_f1"])]:
    for bar, v in zip(bars, vals):
        ax1.annotate(f"{v:.3f}",
                    (bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

# Significance asterisks
for i, row in df.iterrows():
    if row["significant_05"]:
        ax1.annotate("*",
                    (i, max(row["fedavg_mean_f1"], row["fedprox_mean_f1"])
                        + max(row["fedavg_sd_f1"], row["fedprox_sd_f1"]) + 0.03),
                    ha="center", fontsize=20, color=C_GAIN, fontweight="bold")

labels = [s.split(" ")[0] + "\n(" + ", ".join(r.split(", ")[:2])[:24] + ")"
          for s, r in zip(df["specialty"], df["classes"])]
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Test F1 on specialty classes (mean ± SD across 10 seeds)")
ax1.set_ylim(0, 1.05)
ax1.set_title("(A) Per-specialty test F1\n"
              "* = significant at p < 0.05 (paired Wilcoxon)")
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25, axis="y")
ax1.spines[["top", "right"]].set_visible(False)

# Right panel: Δ with significance and win rate
deltas    = df["delta_mean"].values
delta_sds = df["delta_sd"].values
colors    = [C_GAIN if s else "#a0aec0" for s in df["significant_05"]]
bars = ax2.barh(np.arange(len(df)), deltas, xerr=delta_sds, capsize=5,
                color=colors, edgecolor="black")
for i, row in df.iterrows():
    sig = " *" if row["significant_05"] else ""
    ax2.annotate(f"{row['delta_mean']:+.4f}  ({int(row['fedprox_wins'])}/{int(row['n_pairs'])} wins, p={row['wilcoxon_p']:.3f}){sig}",
                xy=(row["delta_mean"] + 0.005, i),
                va="center", ha="left", fontsize=9, fontweight="bold")
ax2.axvline(0, color="black", linewidth=1)
ax2.set_yticks(np.arange(len(df)))
ax2.set_yticklabels([s.split(" ")[0] for s in df["specialty"]], fontsize=10)
ax2.set_xlabel("Δ F1 (FedProx − FedAvg)")
ax2.set_xlim(-0.02, max(deltas) + 0.12)
ax2.set_title("(B) FedProx advantage per specialty")
ax2.grid(alpha=0.25, axis="x")
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("Per-specialty analysis — FedProx vs FedAvg on the classes each client pair holds",
             fontsize=12, fontweight="bold")

out_png = FIG_DIR / "12_per_client_specialty.png"
out_pdf = FIG_DIR / "12_per_client_specialty.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf,           bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")
