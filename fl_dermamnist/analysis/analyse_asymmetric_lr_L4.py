"""Experiment D1 — Asymmetric LR analysis (NOVEL Li 2020 §5.2 extension).

Reads 27 runs at
  fl_dermamnist/results/asymmetric_lr_L4/
       test_at_best_{fedavg,fedprox,fednova}_..._lrPC-c0lr0.01-c1lr{0.005,0.002}_s{42,123,456}.json
       test_at_best_{fedavg,fedprox,fednova}_mu0.X_E20_s{42,123,456}.json  (LR=1:1 baseline)

Tests whether the FedProx-vs-FedNova mechanism distinction extends
to asymmetric LR — a regime where Wang 2020's FedNova theory does
NOT apply (FedNova proves correction for unequal tau_i, not LR).

Three LR ratios × 3 algorithms × 3 seeds = 27 runs total.

Predicted (novel if confirmed):
  - FedProx absorbs LR asymmetry via proximal anchor: ~same as 1:1
  - FedNova collapses under LR asymmetry: outside its proved regime
  - FedAvg sits in between

Output:
  - asymmetric_lr_L4_summary.csv
  - F_asymmetric_lr_L4.{pdf,png}  : 2-panel (macro-F1 + rare-class)
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
IN_DIR = REPO_ROOT / "fl_dermamnist/results/asymmetric_lr_L4"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]
RARE_IDX = (3, 4, 6)


def _scan() -> pd.DataFrame:
    """Read all JSONs and parse LR pair from the lr_per_client metadata."""
    rows = []
    files = list(IN_DIR.glob("test_at_best_*.json"))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        algo = d.get("algorithm")
        seed = d.get("seed")
        # Reconstruct the LR pair
        lr_pc = d.get("lr_per_client")
        global_lr = d.get("lr")
        if lr_pc is not None and isinstance(lr_pc, dict):
            c0_lr = float(lr_pc.get("0", global_lr))
            c1_lr = float(lr_pc.get("1", global_lr))
        else:
            c0_lr = float(global_lr); c1_lr = float(global_lr)
        # LR ratio C0:C1 reduced
        ratio_str = f"{c0_lr}:{c1_lr}"
        # Classify the asymmetry level
        if abs(c0_lr - c1_lr) < 1e-9:
            ratio_label = "1:1 (symmetric)"
            ratio_order = 0
        elif abs(c1_lr - c0_lr / 2) < 1e-9:
            ratio_label = "2:1"
            ratio_order = 1
        elif abs(c1_lr - c0_lr / 5) < 1e-9:
            ratio_label = "5:1"
            ratio_order = 2
        else:
            ratio_label = ratio_str
            ratio_order = 99
        pc = d.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            algorithm=algo,
            seed=seed,
            c0_lr=c0_lr, c1_lr=c1_lr,
            ratio_label=ratio_label,
            ratio_order=ratio_order,
            macro_f1=d.get("macro_f1"),
            balanced_accuracy=d.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
            file=f.name,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------
df = _scan()
if df.empty:
    print(f"WARNING: no result files in {IN_DIR}. Submit jobs first.")
    raise SystemExit(0)
df = df.sort_values(["algorithm", "ratio_order", "seed"])
df.to_csv(OUT_DIR / "asymmetric_lr_L4_summary.csv", index=False)
print(f"Wrote {OUT_DIR/'asymmetric_lr_L4_summary.csv'}  ({len(df)} runs)")
print()
print(df[["algorithm", "ratio_label", "seed", "macro_f1", "rare_avg_f1"]].to_string(index=False))

# ----------------------------------------------------------------
# 2. 3-seed mean +/- SD per (algorithm, ratio)
# ----------------------------------------------------------------
summary = df.groupby(["algorithm", "ratio_label", "ratio_order"]).agg(
    n=("macro_f1", "count"),
    mean_macro=("macro_f1", "mean"),
    sd_macro=("macro_f1", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
    mean_rare=("rare_avg_f1", "mean"),
    sd_rare=("rare_avg_f1", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
).reset_index().sort_values(["algorithm", "ratio_order"])
print()
print("=" * 80)
print("3-seed mean +/- SD per (algorithm, LR ratio)")
print("=" * 80)
print(summary.to_string(index=False))

# ----------------------------------------------------------------
# 3. Compute "LR-asymmetry penalty" per algorithm
# ----------------------------------------------------------------
print()
print("=" * 80)
print("LR-asymmetry penalty: drop from symmetric (1:1) to most asymmetric (5:1)")
print("=" * 80)
penalties = []
for algo in ["fedavg", "fedprox", "fednova"]:
    sub = summary[summary["algorithm"] == algo]
    if not len(sub):
        continue
    sym = sub[sub["ratio_order"] == 0]
    asy = sub[sub["ratio_order"] == 2]
    if not len(sym) or not len(asy):
        print(f"  {algo:<10s}: insufficient data")
        continue
    drop = float(sym["mean_macro"].iloc[0] - asy["mean_macro"].iloc[0])
    rare_drop = float(sym["mean_rare"].iloc[0] - asy["mean_rare"].iloc[0])
    penalties.append(dict(algorithm=algo, drop_macro=drop, drop_rare=rare_drop))
    interpretation = "absorbs" if drop < 0.02 else ("affected" if drop < 0.05 else "COLLAPSES")
    print(f"  {algo:<10s}: macro-F1 drop = {drop:+.4f}  rare drop = {rare_drop:+.4f}   --> {interpretation}")

# ----------------------------------------------------------------
# 4. Headline verdict
# ----------------------------------------------------------------
print()
print("=" * 80)
print("PREDICTED vs OBSERVED:")
print("=" * 80)
pen_df = pd.DataFrame(penalties)
if len(pen_df) == 3:
    fa = pen_df[pen_df["algorithm"] == "fedavg"]["drop_macro"].iloc[0]
    fp = pen_df[pen_df["algorithm"] == "fedprox"]["drop_macro"].iloc[0]
    fn = pen_df[pen_df["algorithm"] == "fednova"]["drop_macro"].iloc[0]
    print(f"  Predicted: FedProx absorbs (drop ~ 0)")
    print(f"             FedAvg   affected")
    print(f"             FedNova  collapses (>= 0.10 drop)")
    print()
    print(f"  Observed:  FedAvg   {fa:+.4f}")
    print(f"             FedProx  {fp:+.4f}")
    print(f"             FedNova  {fn:+.4f}")
    print()
    if fp < 0.02 and fn > fp + 0.05:
        print("  --> CONFIRMED: FedProx-vs-FedNova mechanism distinction "
              "extends to LR asymmetry")
    elif abs(fp - fn) < 0.02:
        print("  --> NULL: FedProx and FedNova behave similarly under LR asymmetry")
    else:
        print("  --> MIXED: see per-condition numbers for nuance")

# ----------------------------------------------------------------
# 5. Figure: 2-panel — macro-F1 per (algo, ratio) + rare-class
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

COLORS = {"fedavg": "#7FBF94", "fedprox": "#3D5A80", "fednova": "#C9A227"}
RATIO_ORDER = ["1:1 (symmetric)", "2:1", "5:1"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.25})

# Panel A: macro-F1 vs LR ratio, one line per algorithm
x = np.arange(len(RATIO_ORDER))
w = 0.27
for i, algo in enumerate(["fedavg", "fedprox", "fednova"]):
    sub = summary[summary["algorithm"] == algo].sort_values("ratio_order")
    means = []
    sds = []
    for r in RATIO_ORDER:
        s = sub[sub["ratio_label"] == r]
        if len(s):
            means.append(float(s["mean_macro"].iloc[0]))
            sds.append(float(s["sd_macro"].iloc[0]))
        else:
            means.append(np.nan); sds.append(0.0)
    axA.bar(x + (i - 1) * w, means, w, yerr=sds,
            color=COLORS[algo], edgecolor="white", linewidth=0.6, capsize=4,
            error_kw=dict(linewidth=1, ecolor="#333"),
            label={"fedavg":"FedAvg",
                   "fedprox":r"FedProx ($\mu=0.01$)",
                   "fednova":"FedNova"}[algo])
    for j, (mv, sv) in enumerate(zip(means, sds)):
        if not np.isnan(mv):
            axA.text(x[j] + (i - 1) * w, mv + sv + 0.005,
                     f"{mv:.3f}", ha="center", va="bottom",
                     fontsize=8.5, color=COLORS[algo])
axA.set_xticks(x); axA.set_xticklabels(RATIO_ORDER, fontsize=10)
axA.set_xlabel("LR asymmetry ratio  (Client 0 : Client 1)")
axA.set_ylabel("Test macro-F1 (mean +/- SD, 3 seeds)")
axA.set_title("(a) Macro-F1 vs LR asymmetry — does FedProx absorb where FedNova collapses?",
              loc="left", fontweight="bold", fontsize=10.5)
axA.legend(loc="upper right", frameon=False, fontsize=9)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: rare-class F1 vs LR ratio
for i, algo in enumerate(["fedavg", "fedprox", "fednova"]):
    sub = summary[summary["algorithm"] == algo].sort_values("ratio_order")
    means = []
    sds = []
    for r in RATIO_ORDER:
        s = sub[sub["ratio_label"] == r]
        if len(s):
            means.append(float(s["mean_rare"].iloc[0]))
            sds.append(float(s["sd_rare"].iloc[0]))
        else:
            means.append(np.nan); sds.append(0.0)
    axB.bar(x + (i - 1) * w, means, w, yerr=sds,
            color=COLORS[algo], edgecolor="white", linewidth=0.6, capsize=4,
            error_kw=dict(linewidth=1, ecolor="#333"),
            label=algo)
axB.set_xticks(x); axB.set_xticklabels(RATIO_ORDER, fontsize=10)
axB.set_xlabel("LR asymmetry ratio")
axB.set_ylabel("Mean rare-class F1 (mean +/- SD, 3 seeds)")
axB.set_title("(b) Rare-class F1 vs LR asymmetry (avg over dermato, melanoma, vascular)",
              loc="left", fontweight="bold", fontsize=10.5)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Asymmetric-LR extension of Li 2020 §5.2 — novel test of FedProx vs FedNova mechanism "
             "under a regime FedNova was NOT proved for",
             fontsize=11, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_asymmetric_lr_L4.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_asymmetric_lr_L4.pdf'}")
print()
print("Done.")
