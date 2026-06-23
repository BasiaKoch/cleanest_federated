"""Extended-rounds 3-seed L3 — convergence-truncation analysis.

Reads the 6 runs at
  fl_dermamnist/results/extended_rounds_L3/test_at_best_{fedavg,fedprox}_mu*_E20_s{42,123,456}.json
and compares them to the original 150-round L3 single-seed run at
  fl_dermamnist/results/heterogeneity_ladder/L3_two_client_70_30_rare_enriched/

Decides whether the original L3 FedProx deficit was an under-training
artefact:
  - Selected round  >  220   → still not plateaued, needs even more rounds
  - Selected round  ≤  150   → 150 was enough, the deficit is real
  - Selected round  in (150, 220] → 250 was needed, original was truncated

Emits a figure showing val_macro_f1 trajectories for all 6 runs (3 seeds
× 2 algos), overlaid with the original 150-round run shown as a dashed line.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
IN_DIR = REPO_ROOT / "fl_dermamnist/results/extended_rounds_L3"
ORIG_DIR = REPO_ROOT / "fl_dermamnist/results/heterogeneity_ladder/L3_two_client_70_30_rare_enriched"
OUT_DIR = IN_DIR / "analysis"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
ALGOS = [("fedavg", "0.0"), ("fedprox", "0.01")]
CLASS_NAMES = [
    "actinic", "basal", "benign_kerat", "dermato", "melanoma", "mel_nevi", "vascular",
]
RARE_IDX = (3, 4, 6)

# --- 1) Load extended runs ---
rows = []
for algo, mu in ALGOS:
    for seed in SEEDS:
        f = IN_DIR / f"test_at_best_{algo}_mu{mu}_E20_s{seed}.json"
        if not f.exists():
            print(f"  WARN: missing {f}")
            continue
        x = json.load(open(f))
        pc = x.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            algorithm=algo, mu=float(mu), seed=seed,
            hostname=x.get("hostname", "?"),
            num_rounds=x.get("num_rounds", 250),
            selected_round=x.get("selected_round"),
            best_val_macro_f1=x.get("best_val_macro_f1"),
            macro_f1=x.get("macro_f1"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "extended_rounds_L3_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'extended_rounds_L3_summary.csv'}  ({len(df)} runs)")
print()
print(df[["algorithm", "seed", "num_rounds", "selected_round",
         "best_val_macro_f1", "macro_f1", "rare_avg_f1"]].to_string(index=False))

# --- 2) Convergence-truncation verdict ---
print()
print("=" * 70)
print("Convergence verdict (per algorithm)")
print("=" * 70)
for algo, _ in ALGOS:
    sub = df[df["algorithm"] == algo]
    if not len(sub):
        continue
    rounds = list(sub["selected_round"])
    mean_round = float(sub["selected_round"].mean())
    max_round  = int(sub["selected_round"].max())
    if max_round > 220:
        verdict = "STILL not plateaued — DermaMNIST L3 may need >250 rounds."
    elif max_round <= 150:
        verdict = "150 rounds was sufficient; the original deficit was real."
    else:
        verdict = "Plateau falls in (150, 220]; original 150-round run was TRUNCATED."
    print(f"  {algo:>8}:  selected_rounds = {rounds}  (mean = {mean_round:.0f}, max = {max_round})")
    print(f"            → {verdict}")

# --- 3) Compare to the original 150-round L3 result ---
print()
print("=" * 70)
print("New (extended, 3 seeds) vs original (single-seed, 150 rounds)")
print("=" * 70)
for algo, mu in ALGOS:
    new_sub = df[df["algorithm"] == algo]
    if not len(new_sub):
        continue
    new_mean = float(new_sub["macro_f1"].mean())
    new_sd   = float(new_sub["macro_f1"].std(ddof=1)) if len(new_sub) > 1 else 0.0
    orig_f = ORIG_DIR / f"test_at_best_{algo}_mu{mu}_E20_s42.json"
    if orig_f.exists():
        orig = json.load(open(orig_f))
        orig_macro = orig.get("macro_f1")
        orig_round = orig.get("selected_round")
        delta = new_mean - orig_macro
        print(f"  {algo:>8}:  extended (3 seeds) = {new_mean:.4f} ± {new_sd:.4f}  "
              f"|  original (s42, 150r, sel_round={orig_round}) = {orig_macro:.4f}  "
              f"|  Δ = {delta:+.4f}")
    else:
        print(f"  {algo:>8}:  extended = {new_mean:.4f} ± {new_sd:.4f}  |  original = MISSING")

# --- 4) Figure: val_macro_f1 trajectories overlay ---
COLOR_FA = "#7FBF94"
COLOR_FP = "#3D5A80"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

# Plot extended runs (solid lines).
for algo, mu in ALGOS:
    color = COLOR_FA if algo == "fedavg" else COLOR_FP
    for seed in SEEDS:
        hist_f = IN_DIR / f"history_{algo}_mu{mu}_E20_s{seed}.csv"
        if not hist_f.exists():
            continue
        hist = pd.read_csv(hist_f)
        ax.plot(hist["round"], hist["val_macro_f1"],
                color=color, linewidth=1.1, alpha=0.55,
                label=f"{algo} s{seed}" if seed == SEEDS[0] else None)

# Overlay original 150-round runs (dashed, darker).
for algo, mu in ALGOS:
    color = COLOR_FA if algo == "fedavg" else COLOR_FP
    orig_hist = ORIG_DIR / f"history_{algo}_mu{mu}_E20_s42.csv"
    if orig_hist.exists():
        hist = pd.read_csv(orig_hist)
        ax.plot(hist["round"], hist["val_macro_f1"],
                color=color, linewidth=2.2, linestyle="--", alpha=0.9,
                label=f"{algo} original 150r")

# Vertical line at round 150 to mark the original cap.
ax.axvline(150, color="#555", linestyle=":", linewidth=1.0, alpha=0.7)
ax.text(150, ax.get_ylim()[1], "  original 150-round cap", fontsize=8,
        color="#555", va="top")

ax.set_xlabel("Round")
ax.set_ylabel("Validation macro-F1")
ax.set_title("L3 (mixed 70/30, JS = 0.12)  —  extended-rounds 3-seed vs original 150-round single-seed",
             loc="left", fontweight="bold", fontsize=11)
ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Two-legend layout: one for line style (3-seed vs original), one for algo colour.
from matplotlib.lines import Line2D
legend_lines = [
    Line2D([0], [0], color=COLOR_FA, linewidth=1.5, label="FedAvg"),
    Line2D([0], [0], color=COLOR_FP, linewidth=1.5, label=r"FedProx ($\mu=0.01$)"),
    Line2D([0], [0], color="#555", linewidth=1.5, linestyle="-", alpha=0.55, label="extended (250r, 3 seeds)"),
    Line2D([0], [0], color="#555", linewidth=2.0, linestyle="--", label="original (150r, seed 42)"),
]
ax.legend(handles=legend_lines, loc="lower right", frameon=False, fontsize=9)

for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_extended_rounds_L3.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_extended_rounds_L3.pdf'}")
print()
print("Done.")
