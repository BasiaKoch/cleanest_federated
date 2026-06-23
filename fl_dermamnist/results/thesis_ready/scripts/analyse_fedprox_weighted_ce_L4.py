"""Experiment B2 — FedProx × class-weighted-CE 2x2 analysis.

Tests whether FedProx (drift damping via proximal term) and
class-weighted CE (loss-side rebalancing) are SUBSTITUTES,
COMPLEMENTS (additive), or NON-ADDITIVE on partition-induced class
imbalance.

Reads 12 runs at
  fl_dermamnist/results/fedprox_weighted_ce_L4/
       test_at_best_{fedavg,fedprox}_mu{0.0,0.01}_E20_s{42,123,456}.json

The 2x2 design:
  (a) FedAvg + standard CE             baseline
  (b) FedAvg + class_weighted_ce       loss-only intervention
  (c) FedProx (mu=0.01) + standard CE  algorithm-only intervention
  (d) FedProx + class_weighted_ce      combined

Decomposition:
  loss_effect      = E[(b) - (a)]       benefit from loss rebalancing alone
  algo_effect      = E[(c) - (a)]       benefit from FedProx alone
  combined_effect  = E[(d) - (a)]       benefit of stacking
  interaction      = combined - (loss_effect + algo_effect)
                     ~ 0  -> additive  (complement)
                     < 0  -> SUB-additive (substitutes — diminishing returns)
                     > 0  -> SUPER-additive (synergistic)

Output:
  - fedprox_weighted_ce_L4_summary.csv
  - F_fedprox_weighted_ce_L4.{pdf,png}  : 2-panel (macro-F1 + rare-class)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
IN_DIR = REPO_ROOT / "fl_dermamnist/results/fedprox_weighted_ce_L4"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]
RARE_IDX = (3, 4, 6)


# Filename pattern derived from run_one_flower.py stem rules.
# loss-type is embedded in the run via --loss-type but NOT in the
# filename stem. To distinguish, we read each JSON's loss_type field.
def _scan() -> pd.DataFrame:
    rows = []
    files = list(IN_DIR.glob("test_at_best_*.json"))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        algo = d.get("algorithm")
        loss = d.get("loss_type", "ce")
        seed = d.get("seed")
        pc = d.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            algorithm=algo,
            loss_type=loss,
            seed=seed,
            macro_f1=d.get("macro_f1"),
            balanced_accuracy=d.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
            file=f.name,
            condition=f"{algo}_{loss}",
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# 1. Load and pivot
# ----------------------------------------------------------------
df = _scan()
if df.empty:
    print(f"WARNING: no result files in {IN_DIR}. Submit jobs first.")
    raise SystemExit(0)
df.to_csv(OUT_DIR / "fedprox_weighted_ce_L4_summary.csv", index=False)
print(f"Wrote {OUT_DIR/'fedprox_weighted_ce_L4_summary.csv'}  ({len(df)} runs)")
print()
print(df[["condition", "seed", "macro_f1", "rare_avg_f1"]].to_string(index=False))

# ----------------------------------------------------------------
# 2. 3-seed mean +/- SD per condition
# ----------------------------------------------------------------
summary = df.groupby("condition").agg(
    n=("macro_f1", "count"),
    mean_macro=("macro_f1", "mean"),
    sd_macro=("macro_f1", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
    mean_rare=("rare_avg_f1", "mean"),
    sd_rare=("rare_avg_f1", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
).reset_index()
print()
print("=" * 80)
print("3-seed mean +/- SD per condition")
print("=" * 80)
print(summary.to_string(index=False))

# ----------------------------------------------------------------
# 3. 2x2 ANOVA-style decomposition
# ----------------------------------------------------------------
def _g(cond):
    sub = df[df["condition"] == cond]
    if not len(sub):
        return np.nan, 0
    return float(sub["macro_f1"].mean()), len(sub)

a, _ = _g("fedavg_ce")              # baseline
b, _ = _g("fedavg_class_weighted_ce")
c, _ = _g("fedprox_ce")
d, _ = _g("fedprox_class_weighted_ce")

print()
print("=" * 80)
print("2x2 decomposition (macro-F1):")
print("=" * 80)
print(f"  (a) FedAvg + CE                 : {a:.4f}")
print(f"  (b) FedAvg + weighted-CE        : {b:.4f}")
print(f"  (c) FedProx + CE                : {c:.4f}")
print(f"  (d) FedProx + weighted-CE       : {d:.4f}")
print()
loss_effect = b - a
algo_effect = c - a
combined = d - a
interaction = combined - (loss_effect + algo_effect)
print(f"  loss_effect       = (b)-(a)     = {loss_effect:+.4f}")
print(f"  algo_effect       = (c)-(a)     = {algo_effect:+.4f}")
print(f"  combined_effect   = (d)-(a)     = {combined:+.4f}")
print(f"  expected_additive = loss+algo   = {loss_effect+algo_effect:+.4f}")
print(f"  interaction       = combined - expected = {interaction:+.4f}")
print()
if abs(interaction) < 0.005:
    verdict = "ADDITIVE (complements; effects stack cleanly)"
elif interaction < -0.005:
    verdict = "SUB-ADDITIVE (substitutes; diminishing returns from stacking)"
else:
    verdict = "SUPER-ADDITIVE (synergy)"
print(f"  Verdict: {verdict}")

# ----------------------------------------------------------------
# 4. Figure: 2-panel — macro-F1 dot plot + per-class breakdown
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

# Condition ordering for plotting
COND_ORDER = ["fedavg_ce", "fedavg_class_weighted_ce",
              "fedprox_ce", "fedprox_class_weighted_ce"]
COND_LABEL = {
    "fedavg_ce":                    "(a) FedAvg + CE\n(baseline)",
    "fedavg_class_weighted_ce":     "(b) FedAvg + weighted-CE\n(loss-only)",
    "fedprox_ce":                   "(c) FedProx + CE\n(algo-only)",
    "fedprox_class_weighted_ce":    "(d) FedProx + weighted-CE\n(combined)",
}
COND_COLORS = {"fedavg_ce": "#7FBF94", "fedavg_class_weighted_ce": "#C9A227",
               "fedprox_ce": "#3D5A80", "fedprox_class_weighted_ce": "#C03A2B"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.25})

# Panel A: macro-F1 dot plot
present = [c for c in COND_ORDER if c in df["condition"].values]
for j, cond in enumerate(present):
    sub = df[df["condition"] == cond]
    color = COND_COLORS.get(cond, "#666")
    xs = np.full(len(sub), j) + np.random.RandomState(j).uniform(-0.08, 0.08, size=len(sub))
    axA.scatter(xs, sub["macro_f1"], c=color, s=90, edgecolor="white",
                linewidth=0.8, zorder=3)
    mean = float(sub["macro_f1"].mean())
    sd = float(sub["macro_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
    axA.errorbar([j], [mean], yerr=[sd], fmt="_", color=color,
                 markersize=30, linewidth=2, capsize=10, zorder=2)
    axA.text(j, mean + sd + 0.005, f"{mean:.3f}\n+/-{sd:.3f}",
             ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
axA.set_xticks(range(len(present)))
axA.set_xticklabels([COND_LABEL[c] for c in present], fontsize=8.5)
axA.set_ylabel("Test macro-F1")
axA.set_title(f"(a) 2x2 macro-F1; interaction = {interaction:+.4f} ({verdict.split('(')[0].strip()})",
              loc="left", fontweight="bold", fontsize=10.5)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: per-class F1
x = np.arange(7)
group_w = 0.85
bar_w = group_w / max(len(present), 1)
for j, cond in enumerate(present):
    sub = df[df["condition"] == cond]
    color = COND_COLORS.get(cond, "#666")
    means = [sub[f"f1_{CLASS_NAMES[c]}"].mean() for c in range(7)]
    sds = [sub[f"f1_{CLASS_NAMES[c]}"].std(ddof=1) if len(sub) > 1 else 0.0
           for c in range(7)]
    offset = (j - (len(present) - 1) / 2) * bar_w
    axB.bar(x + offset, means, bar_w, yerr=sds,
            color=color, edgecolor="white", linewidth=0.4, capsize=2,
            error_kw=dict(linewidth=0.8, ecolor="#333"),
            label=COND_LABEL[cond].replace("\n", " "))
for c in RARE_IDX:
    axB.axvspan(c - 0.5, c + 0.5, color="#C9A227", alpha=0.10)
axB.set_xticks(x); axB.set_xticklabels(CLASS_NAMES, rotation=20, ha="right",
                                      fontsize=8.5)
axB.set_ylabel("Per-class test F1 (mean +/- SD, 3 seeds)")
axB.set_title("(b) Per-class F1 — rare classes shaded",
              loc="left", fontweight="bold", fontsize=11)
axB.legend(loc="upper left", frameon=False, fontsize=7.5)
axB.set_ylim(0, 1.0)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("FedProx x class-weighted-CE compositionality on L4 — "
             "tests substitution vs complementarity of drift damping vs loss rebalancing",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_fedprox_weighted_ce_L4.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_fedprox_weighted_ce_L4.pdf'}")
print()
print("Done.")
