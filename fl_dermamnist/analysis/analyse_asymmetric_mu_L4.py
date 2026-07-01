"""Experiment B - Asymmetric per-client μ on L4 analysis.

Reads the 12 runs at
  fl_dermamnist/results/asymmetric_mu_L4/
       test_at_best_{algo}_mu0.01[_muPC-...]_E20_s{42,123,456}.json

Decomposes 4 conditions × 3 seeds:
  1. FedAvg                       (μ_0 = 0,    μ_1 = 0)
  2. Symmetric FedProx            (μ_0 = 0.01, μ_1 = 0.01)
  3. ⭐ Asymmetric anchor-large    (μ_0 = 0.01, μ_1 = 0)
  4. Asymmetric anchor-small CTRL (μ_0 = 0,    μ_1 = 0.01)

Headline question: does setting μ = 0 on the small specialist client
(condition 3) recover the vascular-class F1 that symmetric FedProx
(condition 2) collapses?

Theoretical anchor: Yao et al. 2024 (NeurIPS, arXiv:2410.08934)
predicts per-client μ should track per-client heterogeneity.
Client 1 has the highest heterogeneity (class-disjoint, all-rare-classes),
so Yao's prediction is that μ_1 should be SMALL (zero or near-zero).

Outputs:
  - asymmetric_mu_L4_summary.csv  (long format with 3-seed mean ± SD)
  - F_asymmetric_mu_L4.{pdf,png}  (2-panel: per-condition macro-F1 + per-class)
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
IN_DIR = REPO_ROOT / "fl_dermamnist/results/asymmetric_mu_L4"
OUT_DIR = IN_DIR / "analysis"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
CLASS_NAMES = [
    "actinic", "basal", "benign_kerat", "dermato", "melanoma", "mel_nevi", "vascular",
]
RARE_IDX = (3, 4, 6)

# (label, filename pattern, ordering index)
CONDITIONS = [
    ("FedAvg",          "fedavg_mu0.0_E20",                      0),
    ("FedProx symmetric (μ=0.01 both)",      "fedprox_mu0.01_E20",                     1),
    ("⭐ Anchor-large (μ_0=0.01, μ_1=0)",     "fedprox_mu0.01_E20_muPC-c0m0.01-c1m0.0", 2),
    ("Anchor-small CTRL (μ_0=0, μ_1=0.01)",  "fedprox_mu0.01_E20_muPC-c0m0.0-c1m0.01", 3),
]


def _read(seed: int, stem: str) -> dict | None:
    f = IN_DIR / f"test_at_best_{stem}_s{seed}.json"
    if not f.exists():
        return None
    return json.load(open(f))


rows = []
for label, stem, idx in CONDITIONS:
    for seed in SEEDS:
        x = _read(seed, stem)
        if x is None:
            print(f"  WARN: missing s{seed}/{stem}")
            continue
        pc = x.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            condition=label, cond_idx=idx, seed=seed,
            hostname=x.get("hostname", "?"),
            selected_round=x.get("selected_round"),
            macro_f1=x.get("macro_f1"),
            balanced_accuracy=x.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "asymmetric_mu_L4_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'asymmetric_mu_L4_summary.csv'}  ({len(df)} runs)")
print()
print(df[["condition", "seed", "macro_f1", "f1_vascular", "rare_avg_f1"]].to_string(index=False))

# --- 3-seed mean ± SD per condition ---
print()
print("=" * 92)
print("3-seed mean ± SD per condition (L4, two_client_90_10_rare_stress)")
print("=" * 92)
summary = []
for label, _, idx in CONDITIONS:
    sub = df[df["condition"] == label]
    if not len(sub):
        continue
    n = len(sub)
    macro_mean = float(sub["macro_f1"].mean())
    macro_sd   = float(sub["macro_f1"].std(ddof=1)) if n > 1 else 0.0
    vasc_mean  = float(sub["f1_vascular"].mean())
    vasc_sd    = float(sub["f1_vascular"].std(ddof=1)) if n > 1 else 0.0
    rare_mean  = float(sub["rare_avg_f1"].mean())
    rare_sd    = float(sub["rare_avg_f1"].std(ddof=1)) if n > 1 else 0.0
    summary.append(dict(condition=label, cond_idx=idx, n=n,
                        macro_mean=macro_mean, macro_sd=macro_sd,
                        vasc_mean=vasc_mean,   vasc_sd=vasc_sd,
                        rare_mean=rare_mean,   rare_sd=rare_sd))
    print(f"  {label:<45} n={n}  macro={macro_mean:.4f}±{macro_sd:.4f}  "
          f"vasc={vasc_mean:.4f}±{vasc_sd:.4f}  rare-avg={rare_mean:.4f}±{rare_sd:.4f}")

# --- Decision rule ---
print()
print("=" * 92)
print("Decision rule")
print("=" * 92)
sumdf = pd.DataFrame(summary)
sym = sumdf[sumdf["condition"].str.startswith("FedProx symmetric")]
asy = sumdf[sumdf["condition"].str.startswith("⭐ Anchor-large")]
ctl = sumdf[sumdf["condition"].str.startswith("Anchor-small CTRL")]
if len(sym) and len(asy):
    s = sym.iloc[0]; a = asy.iloc[0]
    # 1. Does anchor-large recover vascular vs symmetric?
    d_vasc = a["vasc_mean"] - s["vasc_mean"]
    pooled_vsd = np.sqrt(0.5 * (a["vasc_sd"]**2 + s["vasc_sd"]**2))
    print(f"  Vascular F1, anchor-large minus symmetric  = {d_vasc:+.4f}  "
          f"(pooled SD = {pooled_vsd:.4f}, threshold = 2·SD = {2*pooled_vsd:.4f})")
    print(f"    → anchor-large {'RECOVERS' if d_vasc > 2*pooled_vsd else 'DOES NOT recover'} vascular F1")
    # 2. Macro-F1 trade-off
    d_macro = a["macro_mean"] - s["macro_mean"]
    pooled_msd = np.sqrt(0.5 * (a["macro_sd"]**2 + s["macro_sd"]**2))
    print(f"  Macro-F1,  anchor-large minus symmetric  = {d_macro:+.4f}  "
          f"(pooled SD = {pooled_msd:.4f})")
    # 3. Direction matters? Compare anchor-large vs anchor-small (the control).
    if len(ctl):
        c = ctl.iloc[0]
        d_dir = a["vasc_mean"] - c["vasc_mean"]
        pooled_dsd = np.sqrt(0.5 * (a["vasc_sd"]**2 + c["vasc_sd"]**2))
        print(f"  DIRECTION test: anchor-large vs anchor-small vascular F1  = {d_dir:+.4f}  "
              f"(pooled SD = {pooled_dsd:.4f})")
        if d_dir > 2 * pooled_dsd:
            print(f"    → direction MATTERS (Yao 2024 prediction supported on this task)")
        elif abs(d_dir) <= pooled_dsd:
            print(f"    → direction does NOT matter (any reduction in average μ has the effect; "
                  f"undermines the Yao 2024-style 'per-client μ tracks heterogeneity' claim)")
        else:
            print(f"    → directional effect is weak (within 1-2 SD)")

# --- Figure: 2-panel ---
COND_COLORS = ["#7FBF94", "#3D5A80", "#C03A2B", "#C9A227"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.22})

# Panel A: dot plot per condition with mean ± SD bars (macro-F1 + vascular F1)
present = sorted(set(df["cond_idx"]))
for j, idx in enumerate(present):
    sub = df[df["cond_idx"] == idx]
    if not len(sub):
        continue
    label = sub["condition"].iloc[0]
    color = COND_COLORS[idx]
    # macro-F1 dots
    xs = np.full(len(sub), j) + np.random.RandomState(idx).uniform(-0.07, 0.07, size=len(sub))
    axA.scatter(xs, sub["macro_f1"], c=color, s=80, edgecolor="white", linewidth=0.8,
                zorder=3, label=label if idx == 0 else None)
    mean = float(sub["macro_f1"].mean())
    sd   = float(sub["macro_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
    axA.errorbar([j], [mean], yerr=[sd], fmt="_", color=color,
                 markersize=28, linewidth=2, capsize=10, zorder=2)
    axA.text(j, mean + sd + 0.005, f"{mean:.3f}\n±{sd:.3f}",
             ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
axA.set_xticks(range(len(present)))
axA.set_xticklabels([CONDITIONS[i][0] for i in present], rotation=15, ha="right", fontsize=8.5)
axA.set_ylabel("Test macro-F1")
axA.set_title("(a) Macro-F1 across asymmetric-μ conditions (3 seeds)",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: per-class F1 mean ± SD bars, one group per class
x = np.arange(7)
group_w = 0.85
bar_w = group_w / max(len(present), 1)
for j, idx in enumerate(present):
    sub = df[df["cond_idx"] == idx]
    if not len(sub):
        continue
    label = sub["condition"].iloc[0]
    color = COND_COLORS[idx]
    means = [sub[f"f1_{CLASS_NAMES[c]}"].mean() for c in range(7)]
    sds   = [sub[f"f1_{CLASS_NAMES[c]}"].std(ddof=1) if len(sub) > 1 else 0.0 for c in range(7)]
    offset = (j - (len(present) - 1) / 2) * bar_w
    axB.bar(x + offset, means, bar_w, yerr=sds,
            color=color, edgecolor="white", linewidth=0.4, capsize=2,
            error_kw=dict(linewidth=0.8, ecolor="#333"),
            label=label)
for c in RARE_IDX:
    axB.axvspan(c - 0.5, c + 0.5, color="#C9A227", alpha=0.10)
axB.set_xticks(x); axB.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=8.5)
axB.set_ylabel("Per-class test F1 (mean ± SD, 3 seeds)")
axB.set_title("(b) Per-class F1 — rare classes (3, 4, 6) shaded",
              loc="left", fontweight="bold", fontsize=11)
axB.legend(loc="upper left", frameon=False, fontsize=7.5)
axB.set_ylim(0, 1.0)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Asymmetric per-client μ on L4 (two_client_90_10_rare_stress)  —  ablation of Yao et al. 2024 (arXiv:2410.08934)",
             fontsize=11, fontweight="bold", y=1.03)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_asymmetric_mu_L4.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_asymmetric_mu_L4.pdf'}")
print()
print("Done.")
