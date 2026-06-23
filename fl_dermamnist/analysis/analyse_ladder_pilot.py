"""Stage-A ladder pilot analysis (seed 42 only).

Reads results from fl_dermamnist/results/heterogeneity_ladder/L*_*/ and:
  1. Emits a CSV summary (level x algorithm x metrics).
  2. Computes Δ vs FedAvg per level.
  3. Loads the partition-level JS divergence already computed.
  4. Produces the headline ladder figure:
       (a) macro-F1 vs JS divergence (one line per algorithm)
       (b) Δ-macro-F1 vs FedAvg (one bar per (level, alternative))
  5. Produces a per-class rare-class panel.

Output:
  results/heterogeneity_ladder/analysis/
    ladder_summary.csv
    F_ladder_macro_f1.{pdf,png}
    F_ladder_per_class.{pdf,png}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
LADDER_ROOT = REPO_ROOT / "fl_dermamnist/results/heterogeneity_ladder"
OUT_DIR = LADDER_ROOT / "analysis"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

LADDER = [
    ("L0", "two_client_50_50_stratified_iid",          "IID 50/50",          0.0000),
    ("L1", "two_client_86_14_quantity_only_stratified","Quantity 86/14",     0.0000),
    ("L2", "two_client_50_50_label_skew_only",         "Label-skew 50/50",   0.1037),
    ("L3", "two_client_70_30_rare_enriched",           "Mixed 70/30",        0.1206),
    ("L4", "two_client_90_10_rare_stress",             "Severe 86/14",       0.3853),
]

CLASS_NAMES = [
    "actinic", "basal", "benign_kerat", "dermato", "melanoma", "mel_nevi", "vascular",
]
RARE_IDX = (3, 4, 6)

# --- 1) Load all runs ---
rows = []
for level, partition, label, js in LADDER:
    d = LADDER_ROOT / f"{level}_{partition}"
    for f in sorted(d.glob("test_at_best_*.json")):
        x = json.load(open(f))
        algo = x.get("algorithm", "?")
        pc = x.get("per_class_f1") or [float("nan")] * 7
        rare = float(np.mean([pc[i] for i in RARE_IDX])) if len(pc) >= 7 else float("nan")
        rows.append({
            "level": level,
            "partition": partition,
            "label": label,
            "js_divergence": js,
            "algorithm": algo,
            "mu": x.get("mu", 0.0),
            "selected_round": x.get("selected_round"),
            "best_val_macro_f1": x.get("best_val_macro_f1"),
            "macro_f1": x.get("macro_f1"),
            "accuracy": x.get("accuracy"),
            "balanced_accuracy": x.get("balanced_accuracy"),
            "rare_avg_f1": rare,
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
        })
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "ladder_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'ladder_summary.csv'}  ({len(df)} runs)")
print()
print(df[["level", "label", "js_divergence", "algorithm", "macro_f1", "rare_avg_f1"]].to_string(index=False))

# --- 2) Δ vs FedAvg per level ---
delta_rows = []
for (level, label, js), grp in df.groupby(["level", "label", "js_divergence"]):
    if "fedavg" not in grp["algorithm"].values:
        continue
    fa = grp[grp["algorithm"] == "fedavg"].iloc[0]
    for algo in ("fedprox", "fednova"):
        sub = grp[grp["algorithm"] == algo]
        if not len(sub):
            continue
        a = sub.iloc[0]
        delta_rows.append({
            "level": level,
            "label": label,
            "js_divergence": js,
            "method": algo,
            "delta_macro_f1": a["macro_f1"] - fa["macro_f1"],
            "delta_rare_avg_f1": a["rare_avg_f1"] - fa["rare_avg_f1"],
            "delta_balanced_acc": a["balanced_accuracy"] - fa["balanced_accuracy"],
        })
deltas = pd.DataFrame(delta_rows)
deltas.to_csv(OUT_DIR / "ladder_deltas.csv", index=False)
print()
print(deltas.to_string(index=False))

# --- 3) Figure A: macro-F1 vs JS divergence ---
COLORS = {"fedavg": "#7FBF94", "fedprox": "#3D5A80", "fednova": "#C9A227"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5),
                                gridspec_kw={"wspace": 0.22})

# Panel A: raw macro-F1 vs JS
for algo in ("fedavg", "fedprox", "fednova"):
    sub = df[df["algorithm"] == algo].sort_values("js_divergence")
    if not len(sub):
        continue
    axA.plot(sub["js_divergence"], sub["macro_f1"], "-o",
             color=COLORS[algo], linewidth=1.8, markersize=8,
             label={"fedavg":"FedAvg","fedprox":"FedProx ($\\mu=0.01$)","fednova":"FedNova"}[algo])
    for _, row in sub.iterrows():
        axA.annotate(row["level"], (row["js_divergence"], row["macro_f1"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8, color=COLORS[algo])
axA.set_xlabel("Average per-client JS divergence to global label distribution")
axA.set_ylabel("Test macro-F1 (seed 42)")
axA.set_title("(a) Macro-F1 across the heterogeneity ladder",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axA.legend(loc="upper right", frameon=False, fontsize=9)
axA.set_ylim(0.45, 0.65)

# Panel B: Δ vs FedAvg as grouped bars by level
levels = [l[0] for l in LADDER]
x = np.arange(len(levels))
w = 0.32
for i, (algo, color) in enumerate([("fedprox", COLORS["fedprox"]),
                                     ("fednova", COLORS["fednova"])]):
    deltas_for_algo = []
    for lev in levels:
        sub = deltas[(deltas["level"] == lev) & (deltas["method"] == algo)]
        deltas_for_algo.append(float(sub["delta_macro_f1"].iloc[0]) if len(sub) else np.nan)
    bars = axB.bar(x + (i - 0.5)*w, deltas_for_algo, w,
                   color=color, edgecolor="white", linewidth=0.6,
                   label={"fedprox":"FedProx − FedAvg","fednova":"FedNova − FedAvg"}[algo])
    for j, v in enumerate(deltas_for_algo):
        if not np.isnan(v):
            sign = "+" if v >= 0 else ""
            axB.text(x[j] + (i - 0.5)*w, v + (0.001 if v >= 0 else -0.005),
                     f"{sign}{v:.3f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=8.5,
                     color="#1f6f3f" if v > 0 else ("#b04040" if v < 0 else "#555"))
axB.axhline(0, color="#555", linewidth=0.8)
axB.set_xticks(x); axB.set_xticklabels(levels)
axB.set_ylabel("Δ macro-F1 vs FedAvg (seed 42)")
axB.set_title("(b) FedProx / FedNova advantage relative to FedAvg",
              loc="left", fontweight="bold", fontsize=11)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.legend(loc="lower left", frameon=False, fontsize=9)
axB.set_ylim(-0.04, 0.04)

fig.suptitle("Heterogeneity-ladder Stage A pilot (seed 42 only, 1 seed)",
             fontsize=12, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_ladder_macro_f1.{ext}")
plt.close(fig)
print(f"\nWrote {OUT_FIG / 'F_ladder_macro_f1.pdf'}")

# --- 4) Figure B: per-class breakdown across levels for FedAvg / FedProx / FedNova ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True,
                         gridspec_kw={"wspace": 0.07})
for ax, algo in zip(axes, ("fedavg", "fedprox", "fednova")):
    for level, partition, label, js in LADDER:
        sub = df[(df["level"] == level) & (df["algorithm"] == algo)]
        if not len(sub):
            continue
        pc = [sub.iloc[0][f"f1_{nm}"] for nm in CLASS_NAMES]
        ax.plot(range(7), pc, "-o", linewidth=1.4,
                label=f"{level} (JS={js:.2f})")
    ax.set_xticks(range(7))
    ax.set_xticklabels([n.replace("_", "\n") for n in CLASS_NAMES], fontsize=8, rotation=0)
    # Mark rare classes
    for c in RARE_IDX:
        ax.axvspan(c - 0.5, c + 0.5, color="#C9A227", alpha=0.10)
    ax.set_title({"fedavg":"FedAvg","fedprox":"FedProx ($\\mu=0.01$)","fednova":"FedNova"}[algo],
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Per-class test F1 (seed 42)")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False, fontsize=8)
fig.suptitle("Per-class F1 across the heterogeneity ladder — shaded columns = rare classes",
             fontsize=11, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_ladder_per_class.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG / 'F_ladder_per_class.pdf'}")
print()
print("Done.")
