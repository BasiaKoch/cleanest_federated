"""Personalisation-gap analysis for the small-hospital fine-tuning runs.

Reads the 4 fine-tuning JSONs at
  fl_dermamnist/results/small_hospital_finetune/test_at_best_finetune_*.json
and compares pre-FT (= federated global model at best-val) vs post-FT
(= 5 epochs local SGD at lr=0.001 on the client's local data) on the
GLOBAL test set.

Reports per-class deltas and the macro-F1 personalisation gap, and
emits one summary figure.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
FT_DIR = REPO_ROOT / "fl_dermamnist/results/small_hospital_finetune"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "actinic\nkeratoses", "basal cell\ncarcinoma", "benign\nkeratosis",
    "dermatofibroma", "melanoma", "melanocytic\nnevi", "vascular\nlesions",
]
RARE_IDX = (3, 4, 6)

# Configs: 4 runs.
configs = [
    ("fedavg",  0,  "FedAvg + Client 0 FT"),
    ("fedavg",  1,  "FedAvg + Client 1 FT"),
    ("fedprox", 0,  "FedProx + Client 0 FT"),
    ("fedprox", 1,  "FedProx + Client 1 FT"),
]

print("="*90)
print(f"{'Configuration':<28} {'macro pre':<10} {'macro post':<10} {'gap':<10} {'rare pre':<10} {'rare post':<10}")
print("="*90)
records = []
for algo, cid, label in configs:
    mu = "0.0" if algo == "fedavg" else "0.01"
    f = FT_DIR / f"test_at_best_finetune_{algo}_mu{mu}_E20_s42_c{cid}.json"
    d = json.load(open(f))
    pre = np.array(d["pre_ft_per_class_f1"])
    post = np.array(d["per_class_f1"])
    pre_macro = d["pre_ft_macro_f1"]
    post_macro = d["macro_f1"]
    pre_rare = float(np.mean([pre[i] for i in RARE_IDX]))
    post_rare = float(np.mean([post[i] for i in RARE_IDX]))
    gap_macro = post_macro - pre_macro
    print(f"{label:<28} {pre_macro:<10.4f} {post_macro:<10.4f} {gap_macro:<+10.4f} "
          f"{pre_rare:<10.4f} {post_rare:<10.4f}")
    records.append(dict(label=label, algo=algo, cid=cid, pre=pre, post=post,
                        pre_macro=pre_macro, post_macro=post_macro,
                        gap_macro=gap_macro))

# --- Figure: per-class F1 pre vs post FT, four panels ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True,
                          gridspec_kw={"hspace": 0.30, "wspace": 0.08})
axes = axes.flatten()
COL_PRE  = "#7FBF94"   # green = pre-FT (federated)
COL_POST = "#C03A2B"   # red = post-FT
for ax, rec in zip(axes, records):
    x = np.arange(7)
    w = 0.36
    bars_pre  = ax.bar(x - w/2, rec["pre"],  w, color=COL_PRE,  edgecolor="white", linewidth=0.5, label="pre-FT (federated global)")
    bars_post = ax.bar(x + w/2, rec["post"], w, color=COL_POST, edgecolor="white", linewidth=0.5, label="post-FT (after 5 epochs local SGD)")
    for c in RARE_IDX:
        ax.axvspan(c - 0.5, c + 0.5, color="#C9A227", alpha=0.10)
    # Δ annotations
    for i in range(7):
        d = rec["post"][i] - rec["pre"][i]
        sign = "+" if d >= 0 else ""
        col = "#1f6f3f" if d > 0 else ("#b04040" if d < 0 else "#555")
        ymax = max(rec["pre"][i], rec["post"][i])
        ax.text(x[i], ymax + 0.03, f"{sign}{d:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=8, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{rec['label']}  —  macro Δ = {rec['gap_macro']:+.3f}",
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Test F1")
axes[2].set_ylabel("Test F1")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.20),
               ncol=2, frameon=False, fontsize=10)
fig.suptitle("Personalisation-gap analysis: 5-epoch local fine-tuning on the global test set "
             "(seed 42, single seed; rare classes shaded)",
             fontsize=11, fontweight="bold", y=0.995)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_personalisation_gap.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_personalisation_gap.pdf'}")
