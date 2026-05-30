"""μ-sweep across the heterogeneity ladder — Stage A pilot analysis.

Reads the 12 FedProx runs at
  mnist_dermnist/results/mu_sweep_ladder/L{0,2,4}_<partition>/
       test_at_best_fedprox_mu{0.001,0.01,0.1,1.0}_E20_s42.json

and produces:
  1. mu_sweep_summary.csv     — long-format table (level × μ × metrics)
  2. mu_sweep_pivot.csv        — μ × level macro-F1 grid (for the thesis table)
  3. F_mu_sweep_ladder.{pdf,png}
       (a) macro-F1 vs μ, one line per level (log-x); FedAvg shown as dashed reference
       (b) Δ vs FedAvg at each (μ, level), bar grid

A second-stage promotion is justified for any (level, μ*) pair whose
Δ-macro-F1 exceeds the observed cross-node seed-level noise floor
(≈ 0.04 macro-F1).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
SWEEP_ROOT = REPO_ROOT / "mnist_dermnist/results/mu_sweep_ladder"
LADDER_ROOT = REPO_ROOT / "mnist_dermnist/results/heterogeneity_ladder"
OUT_DIR = SWEEP_ROOT / "analysis"
OUT_FIG = REPO_ROOT / "mnist_dermnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Subset of the ladder used for the μ-sweep (3 levels spanning JS range).
LEVELS = [
    ("L0", "two_client_50_50_stratified_iid", "IID 50/50",        0.0000),
    ("L2", "two_client_50_50_label_skew_only", "Label-skew 50/50", 0.1037),
    ("L4", "two_client_90_10_rare_stress",     "Severe 90/10",     0.3853),
]
MU_GRID = [0.001, 0.01, 0.1, 1.0]

CLASS_NAMES = [
    "actinic", "basal", "benign_kerat", "dermato", "melanoma", "mel_nevi", "vascular",
]
RARE_IDX = (3, 4, 6)

# --- 1) Load all 12 FedProx runs from the sweep + the matching FedAvg baselines ---
rows = []
for level, partition, label, js in LEVELS:
    sweep_d = SWEEP_ROOT / f"{level}_{partition}"
    ladder_d = LADDER_ROOT / f"{level}_{partition}"

    # FedAvg baseline: re-use the existing ladder run (no μ).
    fa_path = ladder_d / f"test_at_best_fedavg_mu0.0_E20_s42.json"
    if fa_path.exists():
        x = json.load(open(fa_path))
        pc = x.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            level=level, partition=partition, label=label, js_divergence=js,
            algorithm="fedavg", mu=0.0,
            selected_round=x.get("selected_round"),
            macro_f1=x.get("macro_f1"),
            balanced_accuracy=x.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
        ))
    else:
        print(f"  WARN: missing FedAvg baseline at {fa_path}")

    # FedProx sweep: 4 μ values.
    for mu in MU_GRID:
        fx_path = sweep_d / f"test_at_best_fedprox_mu{mu}_E20_s42.json"
        if not fx_path.exists():
            print(f"  WARN: missing {fx_path}")
            continue
        x = json.load(open(fx_path))
        pc = x.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            level=level, partition=partition, label=label, js_divergence=js,
            algorithm="fedprox", mu=mu,
            selected_round=x.get("selected_round"),
            macro_f1=x.get("macro_f1"),
            balanced_accuracy=x.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
        ))

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "mu_sweep_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'mu_sweep_summary.csv'}  ({len(df)} runs)")
print()
print(df[["level", "label", "js_divergence", "algorithm", "mu",
         "macro_f1", "rare_avg_f1", "selected_round"]].to_string(index=False))

# --- 2) μ × level pivot of macro-F1 (only FedProx) ---
fp = df[df["algorithm"] == "fedprox"].copy()
pivot = fp.pivot_table(index="mu", columns="level", values="macro_f1")
pivot = pivot.reindex(index=MU_GRID, columns=[l[0] for l in LEVELS])
pivot.to_csv(OUT_DIR / "mu_sweep_pivot.csv")
print()
print("Pivot (rows=μ, cols=level), test macro-F1:")
print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

# Best μ per level
print()
print("Best μ per level:")
best_mu = pivot.idxmax(axis=0)
best_val = pivot.max(axis=0)
for lev in pivot.columns:
    fa_macro = float(df[(df["level"] == lev) & (df["algorithm"] == "fedavg")]["macro_f1"].iloc[0])
    delta = best_val[lev] - fa_macro
    print(f"  {lev}:  μ* = {best_mu[lev]:>5}  →  macro-F1 = {best_val[lev]:.4f}  "
          f"(Δ vs FedAvg = {delta:+.4f})")

# --- 3) Figure: μ-sweep curve + Δ bars ---
COLORS_LEVEL = {"L0": "#7FBF94", "L2": "#3D5A80", "L4": "#C03A2B"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5),
                                gridspec_kw={"wspace": 0.25})

# Panel A: macro-F1 vs μ, log-x, one line per level
for level, partition, label, js in LEVELS:
    sub = df[(df["level"] == level) & (df["algorithm"] == "fedprox")].sort_values("mu")
    if not len(sub):
        continue
    axA.plot(sub["mu"], sub["macro_f1"], "-o",
             color=COLORS_LEVEL[level], linewidth=1.8, markersize=8,
             label=f"{level}  ({label}, JS = {js:.3f})")
    # FedAvg baseline (no μ) as a dashed horizontal line in same colour.
    fa = df[(df["level"] == level) & (df["algorithm"] == "fedavg")]
    if len(fa):
        axA.axhline(float(fa["macro_f1"].iloc[0]), color=COLORS_LEVEL[level],
                    linestyle="--", linewidth=1.0, alpha=0.6)
axA.set_xscale("log")
axA.set_xlabel(r"FedProx proximal coefficient $\mu$ (log scale)")
axA.set_ylabel("Test macro-F1 (seed 42)")
axA.set_title(r"(a) FedProx macro-F1 across $\mu$  —  dashed = FedAvg baseline",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, which="both", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axA.legend(loc="lower left", frameon=False, fontsize=9)

# Panel B: Δ vs FedAvg per (μ, level), grouped bars
levels_list = [l[0] for l in LEVELS]
x = np.arange(len(MU_GRID))
w = 0.25
for i, level in enumerate(levels_list):
    fa = df[(df["level"] == level) & (df["algorithm"] == "fedavg")]
    if not len(fa):
        continue
    fa_macro = float(fa["macro_f1"].iloc[0])
    deltas = []
    for mu in MU_GRID:
        sub = df[(df["level"] == level) & (df["algorithm"] == "fedprox") & (df["mu"] == mu)]
        deltas.append(float(sub["macro_f1"].iloc[0]) - fa_macro if len(sub) else np.nan)
    bars = axB.bar(x + (i - 1) * w, deltas, w,
                   color=COLORS_LEVEL[level], edgecolor="white", linewidth=0.6,
                   label=level)
    for j, v in enumerate(deltas):
        if not np.isnan(v):
            sign = "+" if v >= 0 else ""
            axB.text(x[j] + (i - 1) * w, v + (0.002 if v >= 0 else -0.005),
                     f"{sign}{v:.3f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=8,
                     color="#1f6f3f" if v > 0 else ("#b04040" if v < 0 else "#555"))
# Shade the observed seed-level noise floor (±0.04 across HPC nodes on s42).
axB.axhspan(-0.04, 0.04, color="#888", alpha=0.10,
            label="cross-node noise floor (s42)")
axB.axhline(0, color="#555", linewidth=0.8)
axB.set_xticks(x); axB.set_xticklabels([f"$\\mu={m}$" for m in MU_GRID])
axB.set_ylabel("Δ macro-F1 vs FedAvg (seed 42)")
axB.set_title(r"(b) FedProx advantage relative to FedAvg, per $\mu$",
              loc="left", fontweight="bold", fontsize=11)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)

fig.suptitle(r"FedProx $\mu$-sweep Stage A pilot — single seed (42); promote $\mu^*$ candidates to 3-seed Stage B",
             fontsize=12, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_mu_sweep_ladder.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_mu_sweep_ladder.pdf'}")
print()
print("Done.")
